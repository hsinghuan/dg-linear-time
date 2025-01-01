import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from src.models.modules.dygformer import NeighborCooccurrenceEncoder
from src.models.modules.time import (
    CosineTimeEncoder,
    ExpTimeEncoder,
    NoTimeEncoder,
    SineCosineTimeEncoder,
)
from src.utils.analysis import (
    analyze_inter_event_time,
    analyze_target_historical_event_time_diff,
)
from src.utils.data import NeighborSampler


class DyGDecoder(nn.Module):
    """DyGDecoder model."""

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        output_dim: int,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        time_encoding_method: str = "sinusoidal",
        avg_time_diff: float = None,
        median_time_diff: float = None,
        std_time_diff: float = None,
        embed_method: str = "separate",
        add_bos: bool = True,
        inter_event_time: bool = False,
        device: str = "cpu",
    ):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param output_dim: int, dimension of the output embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super().__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        # self.encode_src_dst_separately = encode_src_dst_separately
        self.embed_method = embed_method
        self.add_bos = add_bos
        self.inter_event_time = inter_event_time
        self.device = device

        # print("time encoding method:", time_encoding_method)
        # print("avg_time_diff:", avg_time_diff)
        # print("std_time_diff:", std_time_diff)
        if time_encoding_method == "sinusoidal":
            self.time_encoder = CosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "sinecosine":
            self.time_encoder = SineCosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "exponential":
            self.time_encoder = ExpTimeEncoder(
                time_dim=time_feat_dim,
                median_inter_event_time=median_time_diff,
                parameter_requires_grad=False,
            )
        elif time_encoding_method == "linear":
            self.time_encoder = NoTimeEncoder(mean=avg_time_diff, std=std_time_diff)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
        )

        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(
                    in_features=self.patch_size * self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "edge": nn.Linear(
                    in_features=self.patch_size * self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "time": nn.Linear(  # directly map time difference to channel_embedding_dim
                    in_features=self.patch_size * self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "neighbor_co_occurrence": nn.Linear(
                    in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
            }
        )

        self.num_channels = 4

        if embed_method == "self_cross":
            self.transformers = nn.ModuleList(
                [
                    SelfCrossFormer(
                        attention_dim=self.num_channels * self.channel_embedding_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.transformers = nn.ModuleList(
                [
                    TransformerDecoder(
                        attention_dim=self.num_channels * self.channel_embedding_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=self.output_dim,
            bias=True,
        )

        if self.add_bos:
            self.bos_embedding = nn.Parameter(
                torch.empty(1, self.num_channels * self.channel_embedding_dim), requires_grad=True
            )
            nn.init.normal_(self.bos_embedding)

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        analyze_length: bool = False,
    ):
        """Compute source and destination node temporal embeddings :param src_node_ids: ndarray,
        shape (batch_size, ) :param dst_node_ids: ndarray, shape (batch_size, ) :param
        node_interact_times: ndarray, shape (batch_size, ) :return:"""
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        (
            src_nodes_neighbor_ids_list,
            src_nodes_edge_ids_list,
            src_nodes_neighbor_times_list,
        ) = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=src_node_ids, node_interact_times=node_interact_times
        )

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        (
            dst_nodes_neighbor_ids_list,
            dst_nodes_edge_ids_list,
            dst_nodes_neighbor_times_list,
        ) = self.neighbor_sampler.get_all_first_hop_neighbors(
            node_ids=dst_node_ids, node_interact_times=node_interact_times
        )

        if analyze_length:
            (
                src_avg_time_diffs,
                src_median_time_diffs,
                src_max_time_diffs,
                src_num_temporal_neighbors,
            ) = analyze_target_historical_event_time_diff(
                src_nodes_neighbor_times_list,
                node_interact_times,
                num_neighbors=self.max_input_sequence_length - 1,
            )
            (
                dst_avg_time_diffs,
                dst_median_time_diffs,
                dst_max_time_diffs,
                dst_num_temporal_neighbors,
            ) = analyze_target_historical_event_time_diff(
                dst_nodes_neighbor_times_list,
                node_interact_times,
                num_neighbors=self.max_input_sequence_length - 1,
            )
            src_history_length_analysis = {
                "avg_time_diffs": src_avg_time_diffs,
                "median_time_diffs": src_median_time_diffs,
                "max_time_diffs": src_max_time_diffs,
                "num_temporal_neighbors": src_num_temporal_neighbors,
            }
            dst_history_length_analysis = {
                "avg_time_diffs": dst_avg_time_diffs,
                "median_time_diffs": dst_median_time_diffs,
                "max_time_diffs": dst_max_time_diffs,
                "num_temporal_neighbors": dst_num_temporal_neighbors,
            }

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        (
            src_padded_nodes_neighbor_ids,
            src_padded_nodes_edge_ids,
            src_padded_nodes_neighbor_times,
        ) = self.pad_sequences(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
            nodes_edge_ids_list=src_nodes_edge_ids_list,
            nodes_neighbor_times_list=src_nodes_neighbor_times_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        (
            dst_padded_nodes_neighbor_ids,
            dst_padded_nodes_edge_ids,
            dst_padded_nodes_neighbor_times,
        ) = self.pad_sequences(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
            nodes_edge_ids_list=dst_nodes_edge_ids_list,
            nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
            patch_size=self.patch_size,
            max_input_sequence_length=self.max_input_sequence_length,
        )

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        ) = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        )

        # VERSION 1: Compute separately
        # if self.encode_src_dst_separately:
        if self.embed_method == "separate":
            # get the features of the sequence of source and destination nodes
            # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
            # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
            # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
            (
                src_padded_nodes_neighbor_node_raw_features,
                src_padded_nodes_edge_raw_features,
                src_padded_nodes_neighbor_time_features,
            ) = self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )
            # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
            # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
            # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
            (
                dst_padded_nodes_neighbor_node_raw_features,
                dst_padded_nodes_edge_raw_features,
                dst_padded_nodes_neighbor_time_features,
            ) = self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

            # get the patches for source and destination nodes
            # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
            # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
            # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
            (
                src_patches_nodes_neighbor_node_raw_features,
                src_patches_nodes_edge_raw_features,
                src_patches_nodes_neighbor_time_features,
                src_patches_nodes_neighbor_co_occurrence_features,
            ) = self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

            # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
            # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
            # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
            (
                dst_patches_nodes_neighbor_node_raw_features,
                dst_patches_nodes_edge_raw_features,
                dst_patches_nodes_neighbor_time_features,
                dst_patches_nodes_neighbor_co_occurrence_features,
            ) = self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

            # align the patch encoding dimension
            # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
            src_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
                src_patches_nodes_neighbor_node_raw_features
            )
            src_patches_nodes_edge_raw_features = self.projection_layer["edge"](
                src_patches_nodes_edge_raw_features
            )
            src_patches_nodes_neighbor_time_features = self.projection_layer["time"](
                src_patches_nodes_neighbor_time_features
            )
            src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
                "neighbor_co_occurrence"
            ](src_patches_nodes_neighbor_co_occurrence_features)

            # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
            dst_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
                dst_patches_nodes_neighbor_node_raw_features
            )
            dst_patches_nodes_edge_raw_features = self.projection_layer["edge"](
                dst_patches_nodes_edge_raw_features
            )
            dst_patches_nodes_neighbor_time_features = self.projection_layer["time"](
                dst_patches_nodes_neighbor_time_features
            )
            dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
                "neighbor_co_occurrence"
            ](dst_patches_nodes_neighbor_co_occurrence_features)

            batch_size = len(src_patches_nodes_neighbor_node_raw_features)
            src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
            dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

            # Tensor, shape (batch_size, src_num_patches, num_channels, channel_embedding_dim)
            src_patches_data = torch.stack(
                [
                    src_patches_nodes_neighbor_node_raw_features,
                    src_patches_nodes_edge_raw_features,
                    src_patches_nodes_neighbor_time_features,
                    src_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=2,
            )
            # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            src_patches_data = src_patches_data.reshape(
                batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim
            )

            # Tensor, shape (batch_size, dst_num_patches, num_channels, channel_embedding_dim)
            dst_patches_data = torch.stack(
                [
                    dst_patches_nodes_neighbor_node_raw_features,
                    dst_patches_nodes_edge_raw_features,
                    dst_patches_nodes_neighbor_time_features,
                    dst_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=2,
            )
            # Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
            dst_patches_data = dst_patches_data.reshape(
                batch_size, dst_num_patches, self.num_channels * self.channel_embedding_dim
            )
            # print(f"src_patches_data before transformers has nan? {torch.isnan(src_patches_data).any()}")
            # print(f"dst_patches_data before transformers has nan? {torch.isnan(dst_patches_data).any()}")
            # if torch.isnan(src_patches_data).any():
            #     print(src_patches_data)
            # if torch.isnan(dst_patches_data).any():
            #     print(dst_patches_data)

            if self.add_bos:
                # prepend the bos embedding to each sequence
                patches_bos = self.bos_embedding.view(1, 1, -1).repeat(batch_size, 1, 1)
                src_patches_data = torch.cat([patches_bos, src_patches_data], dim=1)
                dst_patches_data = torch.cat([patches_bos, dst_patches_data], dim=1)
                # make sure that we are indexing the right last token
                offset = 1
            else:
                offset = 0

            # print("src_patches_data:", src_patches_data.shape)
            # print("dst_patches_data", dst_patches_data.shape)

            for transformer in self.transformers:
                src_patches_data = transformer(src_patches_data)
                dst_patches_data = transformer(dst_patches_data)

            # find the patch containing the last non-padding token of src_patches_data and dst_patches_data
            def last_non_zero(arr1d):
                non_zero_indices = np.nonzero(arr1d)[0]
                return non_zero_indices[-1] if non_zero_indices.size > 0 else -1

            # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            src_last_nonpadding_indices = np.apply_along_axis(
                last_non_zero, axis=1, arr=src_padded_nodes_neighbor_ids
            )
            src_last_nonpadding_patch_indices = (
                src_last_nonpadding_indices // self.patch_size + offset
            )
            # print("src_last_nonpadding_indices", src_last_nonpadding_patch_indices)
            src_patches_data = src_patches_data[
                torch.arange(src_patches_data.size(0)), src_last_nonpadding_patch_indices, :
            ]

            # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            dst_last_nonpadding_indices = np.apply_along_axis(
                last_non_zero, axis=1, arr=dst_padded_nodes_neighbor_ids
            )
            dst_last_nonpadding_patch_indices = (
                dst_last_nonpadding_indices // self.patch_size + offset
            )
            dst_patches_data = dst_patches_data[
                torch.arange(dst_patches_data.size(0)), dst_last_nonpadding_patch_indices, :
            ]
            # print("dst_last_nonpadding_indices", dst_last_nonpadding_patch_indices)
            # Tensor, shape (batch_size, output_dim)
            src_node_embeddings = self.output_layer(src_patches_data)
            # Tensor, shape (batch_size, output_dim)
            dst_node_embeddings = self.output_layer(dst_patches_data)

        # VERSION 2: Merge source and destination sequences together and forward them
        elif self.embed_method == "naive_merge":
            # print("bos embedding", self.bos_embedding[:,:5])
            # print("src_nodes_neighbor_ids_list", src_padded_nodes_neighbor_ids)
            # print("dst_nodes_neighbor_ids_list", dst_padded_nodes_neighbor_ids)
            (
                merged_padded_nodes_neighbor_ids,
                merged_padded_nodes_edge_ids,
                merged_padded_nodes_neighbor_times,
                merged_padded_nodes_neighbor_co_occurrence_features,
                merged_src_node_indices,
                merged_dst_node_indices,
            ) = self.merge_src_dst_sequences(
                src_padded_nodes_neighbor_ids,
                src_padded_nodes_edge_ids,
                src_padded_nodes_neighbor_times,
                src_padded_nodes_neighbor_co_occurrence_features,
                dst_padded_nodes_neighbor_ids,
                dst_padded_nodes_edge_ids,
                dst_padded_nodes_neighbor_times,
                dst_padded_nodes_neighbor_co_occurrence_features,
            )
            # print("merged_padded_nodes_neighbor_ids", merged_padded_nodes_neighbor_ids)
            # print("merged_src_node_indices", merged_src_node_indices)
            # print("merged_dst_node_indices", merged_dst_node_indices)
            (
                merged_padded_nodes_neighbor_node_raw_features,
                merged_padded_nodes_edge_raw_features,
                merged_padded_nodes_neighbor_time_features,
            ) = self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=merged_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=merged_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=merged_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

            (
                merged_patches_nodes_neighbor_node_raw_features,
                merged_patches_nodes_edge_raw_features,
                merged_patches_nodes_neighbor_time_features,
                merged_patches_nodes_neighbor_co_occurrence_features,
            ) = self.get_patches(
                padded_nodes_neighbor_node_raw_features=merged_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=merged_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=merged_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=merged_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

            merged_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
                merged_patches_nodes_neighbor_node_raw_features
            )
            merged_patches_nodes_edge_raw_features = self.projection_layer["edge"](
                merged_patches_nodes_edge_raw_features
            )
            merged_patches_nodes_neighbor_time_features = self.projection_layer["time"](
                merged_patches_nodes_neighbor_time_features
            )
            merged_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
                "neighbor_co_occurrence"
            ](merged_patches_nodes_neighbor_co_occurrence_features)

            batch_size = len(merged_patches_nodes_neighbor_node_raw_features)
            merged_num_patches = merged_patches_nodes_neighbor_node_raw_features.shape[1]

            # Tensor, shape (batch_size, merged_num_patches, num_channels, channel_embedding_dim)
            merged_patches_data = torch.stack(
                [
                    merged_patches_nodes_neighbor_node_raw_features,
                    merged_patches_nodes_edge_raw_features,
                    merged_patches_nodes_neighbor_time_features,
                    merged_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=2,
            )

            # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            merged_patches_data = merged_patches_data.reshape(
                batch_size, merged_num_patches, self.num_channels * self.channel_embedding_dim
            )

            if self.add_bos:
                # prepend the bos embedding to the sequence
                patches_bos = self.bos_embedding.view(1, 1, -1).repeat(batch_size, 1, 1)
                merged_patches_data = torch.cat([patches_bos, merged_patches_data], dim=1)
                # make sure that we are indexing the right last token
                offset = 1
            else:
                offset = 0

            for transformer in self.transformers:
                merged_patches_data = transformer(merged_patches_data)

            src_patch_indices, dst_patch_indices = (
                merged_src_node_indices // self.patch_size + offset,
                merged_dst_node_indices // self.patch_size + offset,
            )
            src_patches_data = merged_patches_data[
                torch.arange(merged_patches_data.size(0)), src_patch_indices, :
            ]
            dst_patches_data = merged_patches_data[
                torch.arange(merged_patches_data.size(0)), dst_patch_indices, :
            ]
            # print("src_patch_indices", src_patch_indices)
            # print("dst_patch_indices", dst_patch_indices)

            # Tensor, shape (batch_size, output_dim)
            src_node_embeddings = self.output_layer(src_patches_data)
            # Tensor, shape (batch_size, output_dim)
            dst_node_embeddings = self.output_layer(dst_patches_data)

        # VERSION 3: Use Self Cross Attention
        elif self.embed_method == "self_cross":
            # get the features of the sequence of source and destination nodes
            # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
            # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
            # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
            (
                src_padded_nodes_neighbor_node_raw_features,
                src_padded_nodes_edge_raw_features,
                src_padded_nodes_neighbor_time_features,
            ) = self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )
            # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
            # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
            # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
            (
                dst_padded_nodes_neighbor_node_raw_features,
                dst_padded_nodes_edge_raw_features,
                dst_padded_nodes_neighbor_time_features,
            ) = self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )
            # get the patches for source and destination nodes
            # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
            # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
            # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
            # src_patches_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_num_patches, patch_size * neighbor_co_occurrence_feat_dim)
            # src_patches_nodes_neighbor_times, Tensor, shape (batch_size, src_num_patches)
            (
                src_patches_nodes_neighbor_node_raw_features,
                src_patches_nodes_edge_raw_features,
                src_patches_nodes_neighbor_time_features,
                src_patches_nodes_neighbor_co_occurrence_features,
                src_patches_nodes_neighbor_times,
            ) = self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
                padded_nodes_neighbor_times=torch.from_numpy(src_padded_nodes_neighbor_times).to(
                    self.device
                ),
            )
            # print("src_padded_nodes_neighbor_times", src_padded_nodes_neighbor_times)
            # print("dst_padded_nodes_neighbor_times", dst_padded_nodes_neighbor_times)
            # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
            # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
            # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
            # dst_patches_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_num_patches, patch_size * neighbor_co_occurrence_feat_dim)
            # dst_patches_nodes_neighbor_times, Tensor, shape (batch_size, dst_num_patches)
            (
                dst_patches_nodes_neighbor_node_raw_features,
                dst_patches_nodes_edge_raw_features,
                dst_patches_nodes_neighbor_time_features,
                dst_patches_nodes_neighbor_co_occurrence_features,
                dst_patches_nodes_neighbor_times,
            ) = self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
                padded_nodes_neighbor_times=torch.from_numpy(dst_padded_nodes_neighbor_times).to(
                    self.device
                ),
            )
            # align the patch encoding dimension
            # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
            src_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
                src_patches_nodes_neighbor_node_raw_features
            )
            src_patches_nodes_edge_raw_features = self.projection_layer["edge"](
                src_patches_nodes_edge_raw_features
            )
            src_patches_nodes_neighbor_time_features = self.projection_layer["time"](
                src_patches_nodes_neighbor_time_features
            )
            src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
                "neighbor_co_occurrence"
            ](src_patches_nodes_neighbor_co_occurrence_features)

            # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
            dst_patches_nodes_neighbor_node_raw_features = self.projection_layer["node"](
                dst_patches_nodes_neighbor_node_raw_features
            )
            dst_patches_nodes_edge_raw_features = self.projection_layer["edge"](
                dst_patches_nodes_edge_raw_features
            )
            dst_patches_nodes_neighbor_time_features = self.projection_layer["time"](
                dst_patches_nodes_neighbor_time_features
            )
            dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer[
                "neighbor_co_occurrence"
            ](dst_patches_nodes_neighbor_co_occurrence_features)

            batch_size = len(src_patches_nodes_neighbor_node_raw_features)
            src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
            dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

            # Tensor, shape (batch_size, src_num_patches, num_channels, channel_embedding_dim)
            src_patches_data = torch.stack(
                [
                    src_patches_nodes_neighbor_node_raw_features,
                    src_patches_nodes_edge_raw_features,
                    src_patches_nodes_neighbor_time_features,
                    src_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=2,
            )
            # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            src_patches_data = src_patches_data.reshape(
                batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim
            )

            # Tensor, shape (batch_size, dst_num_patches, num_channels, channel_embedding_dim)
            dst_patches_data = torch.stack(
                [
                    dst_patches_nodes_neighbor_node_raw_features,
                    dst_patches_nodes_edge_raw_features,
                    dst_patches_nodes_neighbor_time_features,
                    dst_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=2,
            )
            # Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
            dst_patches_data = dst_patches_data.reshape(
                batch_size, dst_num_patches, self.num_channels * self.channel_embedding_dim
            )
            # print(f"src_patches_data has nan? {torch.isnan(src_patches_data).any()}")
            # print(f"dst_patches_data has nan? {torch.isnan(dst_patches_data).any()}")
            # if torch.isnan(src_patches_data).any():
            #     print(src_patches_data)
            # if torch.isnan(dst_patches_data).any():
            #     print(dst_patches_data)

            if self.add_bos:
                # prepend the bos embedding to each sequence
                patches_bos = self.bos_embedding.view(1, 1, -1).repeat(batch_size, 1, 1)
                src_patches_data = torch.cat([patches_bos, src_patches_data], dim=1)
                dst_patches_data = torch.cat([patches_bos, dst_patches_data], dim=1)
                # use the smallest possible timestamp of the batch - 1 as the timestamp of bos so every token will attend to bos
                min_timestamp = (
                    min(
                        src_patches_nodes_neighbor_times[:, 0].min(),
                        dst_patches_nodes_neighbor_times[:, 0].min(),
                    )
                    - 1.0
                )
                src_patches_nodes_neighbor_times = torch.cat(
                    [
                        torch.full(
                            size=(batch_size, 1), fill_value=min_timestamp, device=self.device
                        ),
                        src_patches_nodes_neighbor_times,
                    ],
                    dim=1,
                )
                dst_patches_nodes_neighbor_times = torch.cat(
                    [
                        torch.full(
                            size=(batch_size, 1), fill_value=min_timestamp, device=self.device
                        ),
                        dst_patches_nodes_neighbor_times,
                    ],
                    dim=1,
                )
                # print("src_patches_nodes_neighbor_times", src_patches_nodes_neighbor_times)
                # print("dst_patches_nodes_neighbor_times", dst_patches_nodes_neighbor_times)
                offset = 1
            else:
                offset = 0

            # compute source node embeddings by using src_patches_data as the self_seq and dst_patches_data as the cross_seq
            src_hidden_states = src_patches_data
            dst_hidden_states = dst_patches_data
            for transformer in self.transformers:
                src_hidden_states, dst_hidden_states = transformer(
                    src_hidden_states,
                    dst_hidden_states,
                    src_patches_nodes_neighbor_times,
                    dst_patches_nodes_neighbor_times,
                )

            # retrieve src_node_embeddings from src_hidden_states
            # find the patch containing the last non-padding token of src_patches_data and dst_patches_data
            def last_non_zero(arr1d):
                non_zero_indices = np.nonzero(arr1d)[0]
                return non_zero_indices[-1] if non_zero_indices.size > 0 else -1

            # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            src_last_nonpadding_indices = np.apply_along_axis(
                last_non_zero, axis=1, arr=src_padded_nodes_neighbor_ids
            )
            src_last_nonpadding_patch_indices = (
                src_last_nonpadding_indices // self.patch_size + offset
            )
            src_node_embeddings = self.output_layer(
                src_hidden_states[
                    torch.arange(src_hidden_states.size(0)), src_last_nonpadding_patch_indices, :
                ]
            )

            # print("src_padded_nodes_neighbor_ids", src_padded_nodes_neighbor_ids)
            # print("src_last_nonpadding_indices", src_last_nonpadding_indices)
            # print("src_last_nonpadding_patch_indices", src_last_nonpadding_patch_indices)

            # compute destination node embeddings by using dst_patches_data as the self_seq and src_patches_data as the cross_seq
            src_hidden_states = src_patches_data
            dst_hidden_states = dst_patches_data
            for transformer in self.transformers:
                dst_hidden_states, src_hidden_states = transformer(
                    dst_hidden_states,
                    src_hidden_states,
                    dst_patches_nodes_neighbor_times,
                    src_patches_nodes_neighbor_times,
                )

            # retrieve dst_node_embeddings from dst_hidden_states
            # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            dst_last_nonpadding_indices = np.apply_along_axis(
                last_non_zero, axis=1, arr=dst_padded_nodes_neighbor_ids
            )
            dst_last_nonpadding_patch_indices = (
                dst_last_nonpadding_indices // self.patch_size + offset
            )
            dst_node_embeddings = self.output_layer(
                dst_hidden_states[
                    torch.arange(dst_hidden_states.size(0)), dst_last_nonpadding_patch_indices, :
                ]
            )

            # print("dst_padded_nodes_neighbor_ids", dst_padded_nodes_neighbor_ids)
            # print("dst_last_nonpadding_indices", dst_last_nonpadding_indices)
            # print("dst_last_nonpadding_patch_indices", dst_last_nonpadding_patch_indices)

        # print(f"src node embedding has nan? {torch.isnan(src_node_embeddings).any()}")
        # print(f"dst node embedding has nan? {torch.isnan(dst_node_embeddings).any()}")
        # if torch.isnan(src_node_embeddings).any():
        #     print(src_node_embeddings)
        # if torch.isnan(dst_node_embeddings).any():
        #     print(dst_node_embeddings)

        if analyze_length:
            return (
                src_node_embeddings,
                dst_node_embeddings,
                src_history_length_analysis,
                dst_history_length_analysis,
            )
        else:
            return src_node_embeddings, dst_node_embeddings

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        patch_size: int = 1,
        max_input_sequence_length: int = 256,
    ):
        """Pad the sequences for nodes in node_ids that respects the order of time :param node_ids:

        ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for
            nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in
            node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor
            interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert (
            max_input_sequence_length - 1 > 0
        ), "Maximal number of neighbors for each node should be greater than 1!"
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert (
                len(nodes_neighbor_ids_list[idx])
                == len(nodes_edge_ids_list[idx])
                == len(nodes_neighbor_times_list[idx])
            )
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][
                    -(max_input_sequence_length - 1) :
                ]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += patch_size - max_seq_length % patch_size
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            num_neighbors = len(nodes_neighbor_ids_list[idx])
            padded_nodes_neighbor_ids[idx, num_neighbors] = node_ids[idx]
            padded_nodes_edge_ids[idx, num_neighbors] = 0
            padded_nodes_neighbor_times[idx, num_neighbors] = node_interact_times[idx]

            if num_neighbors > 0:
                padded_nodes_neighbor_ids[idx, :num_neighbors] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, :num_neighbors] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, :num_neighbors] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
        time_encoder,
    ):
        """Get node, edge and time features :param node_interact_times: ndarray, shape
        (batch_size,) :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length) :param
        padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length) :param
        time_encoder: TimeEncoder, time encoder :return:"""
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)
        ]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)
        ]

        if not self.inter_event_time:  # use target time - historical edge event time
            padded_nodes_neighbor_time_features = (
                torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times)
                .float()
                .to(self.device)
            )
            # print("padded_nodes_neighbor_time_features", padded_nodes_neighbor_time_features)
            # print("padded_nodes_neighbor_times", padded_nodes_neighbor_times)
            # print("node_interact_times", node_interact_times)
        else:  # use inter-event time
            # Tensor, shape (batch_size, max_seq_length, 1)
            padded_nodes_neighbor_times = torch.from_numpy(padded_nodes_neighbor_times)
            padded_nodes_neighbor_time_features = torch.zeros_like(padded_nodes_neighbor_times).to(
                self.device
            )
            padded_nodes_neighbor_time_features[:, : padded_nodes_neighbor_times.size(-1) - 1] = (
                padded_nodes_neighbor_times[:, 1:] - padded_nodes_neighbor_times[:, :-1]
            )
            padded_nodes_neighbor_time_features[padded_nodes_neighbor_time_features < 0] = 0
            # print("(before) padded_nodes_neighbor_time_features", padded_nodes_neighbor_time_features)
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=padded_nodes_neighbor_time_features
        )

        # padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0
        mask = torch.from_numpy(padded_nodes_neighbor_ids != 0).to(self.device)
        padded_nodes_neighbor_time_features = padded_nodes_neighbor_time_features * (
            mask.unsqueeze(-1)
        )
        # print("(after) padded_nodes_neighbor_time_features", padded_nodes_neighbor_time_features)
        # print(f"padded_nodes_neighbor_time_features has inf? {torch.isinf(padded_nodes_neighbor_time_features).any()}")
        # if torch.isinf(padded_nodes_neighbor_time_features).any():
        #     print(padded_nodes_neighbor_time_features)
        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def get_patches(
        self,
        padded_nodes_neighbor_node_raw_features: torch.Tensor,
        padded_nodes_edge_raw_features: torch.Tensor,
        padded_nodes_neighbor_time_features: torch.Tensor,
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
        patch_size: int = 1,
        padded_nodes_neighbor_times: torch.Tensor = None,
    ):
        """Get the sequence of patches for nodes :param padded_nodes_neighbor_node_raw_features:

        Tensor, shape (batch_size, max_seq_length, node_feat_dim) :param
        padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length,
        time_feat_dim) :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape
        (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim) :param patch_size: int, patch
        size :param padded_nodes_neighbor_times: Tensor, shape (batch_size, max_seq_length)
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim or edge_feat_dim or time_feat_dim)
        (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
            patches_nodes_neighbor_times,
        ) = ([], [], [], [], [])

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx:end_idx, :]
            )
            if padded_nodes_neighbor_times is not None:
                patches_nodes_neighbor_times.append(
                    padded_nodes_neighbor_times[:, start_idx:end_idx]
                )

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * neighbor_co_occurrence_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        if padded_nodes_neighbor_times is not None:
            # Tensor, shape (batch_size, num_patches)
            patches_nodes_neighbor_times = (
                torch.stack(patches_nodes_neighbor_times, dim=1)
                .reshape(batch_size, num_patches, patch_size)
                .max(dim=-1)[0]  # TODO: try .mean()
            )

            return (
                patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features,
                patches_nodes_neighbor_times,
            )
        else:
            return (
                patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features,
            )

    def merge_src_dst_sequences(
        self,
        src_padded_nodes_neighbor_ids: np.ndarray,
        src_padded_nodes_edge_ids: np.ndarray,
        src_padded_nodes_neighbor_times: np.ndarray,
        src_padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
        dst_padded_nodes_neighbor_ids: np.ndarray,
        dst_padded_nodes_edge_ids: np.ndarray,
        dst_padded_nodes_neighbor_times: np.ndarray,
        dst_padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
    ):
        """Merge source and destination sequences by interleaving and sort them based on
        timestamps."""

        # find the patch containing the last non-padding token of src_patches_data and dst_patches_data
        def last_non_zero(arr1d):
            non_zero_indices = np.nonzero(arr1d)[0]
            return non_zero_indices[-1] if non_zero_indices.size > 0 else -1

        # print("src_padded_nodes_neighbor_times", src_padded_nodes_neighbor_times)
        # print("src_padded_nodes_neighbor_ids", src_padded_nodes_neighbor_ids)
        # print("src_padded_nodes_edge_ids", src_padded_nodes_edge_ids)
        # print("src_padded_nodes_neighbor_co_occurrence_features", src_padded_nodes_neighbor_co_occurrence_features[:,:,0])

        # print("dst_padded_nodes_neighbor_times", dst_padded_nodes_neighbor_times)
        # print("dst_padded_nodes_neighbor_ids", dst_padded_nodes_neighbor_ids)
        # print("dst_padded_nodes_edge_ids", dst_padded_nodes_edge_ids)
        # print("dst_padded_nodes_neighbor_co_occurrence_features", dst_padded_nodes_neighbor_co_occurrence_features[:,:,0])

        # find the original src and dst node positions in the merged sequence
        merged_src_node_indices = np.apply_along_axis(
            last_non_zero, axis=-1, arr=src_padded_nodes_neighbor_ids
        )
        merged_dst_node_indices = (
            np.apply_along_axis(last_non_zero, axis=-1, arr=dst_padded_nodes_neighbor_ids)
            + src_padded_nodes_neighbor_ids.shape[1]
        )

        # concatenate src_padded_nodes_neighbor_times and dst_padded_nodes_neighbor_times
        merged_padded_nodes_neighbor_times = np.concatenate(
            (src_padded_nodes_neighbor_times, dst_padded_nodes_neighbor_times), axis=-1
        )
        # make the padded node neighbor times (0s at the end of the arrays) to be np.inf so that they are sorted at the end
        merged_padded_nodes_neighbor_times[
            np.concatenate(
                (src_padded_nodes_neighbor_ids == 0, dst_padded_nodes_neighbor_ids == 0), axis=-1
            )
        ] = np.inf
        # sort the concatenated timestamps and get the indices
        sort_indices = np.argsort(
            merged_padded_nodes_neighbor_times, axis=-1
        )  # np.ndarray, (batch_size, max_seq_length * 2)
        merged_padded_nodes_neighbor_times = np.take_along_axis(
            merged_padded_nodes_neighbor_times, sort_indices, axis=-1
        )  # np.ndarray, (batch_size, max_seq_length * 2)
        # merge nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_co_occurrence_features based on the sorted indices
        merged_padded_nodes_neighbor_ids = np.take_along_axis(
            np.concatenate(
                (src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids), axis=-1
            ),
            sort_indices,
            axis=-1,
        )  # np.ndarray, (batch_size, max_seq_length * 2)
        merged_padded_nodes_edge_ids = np.take_along_axis(
            np.concatenate((src_padded_nodes_edge_ids, dst_padded_nodes_edge_ids), axis=-1),
            sort_indices,
            axis=-1,
        )  # np.ndarray, (batch_size, max_seq_length * 2)
        # print("haha", torch.cat((src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features), dim=1))
        # print("hahaha", torch.from_numpy(sort_indices).to(self.device))
        merged_padded_nodes_neighbor_co_occurrence_features = torch.take_along_dim(
            torch.cat(
                (
                    src_padded_nodes_neighbor_co_occurrence_features,
                    dst_padded_nodes_neighbor_co_occurrence_features,
                ),
                dim=1,
            ),
            torch.from_numpy(sort_indices).unsqueeze(-1).to(self.device),
            dim=1,
        )  # torch.Tensor, (batch_size, max_seq_length * 2, neighbor_co_occurrence_feat_dim)

        # find the new src and dst node positions in the sorted sequence
        merged_src_node_indices = np.argmax(
            sort_indices == merged_src_node_indices[:, None], axis=-1
        )
        merged_dst_node_indices = np.argmax(
            sort_indices == merged_dst_node_indices[:, None], axis=-1
        )

        # set the infinity node neighbor times back to 0
        merged_padded_nodes_neighbor_times[merged_padded_nodes_neighbor_times == np.inf] = 0

        # print("merged_padded_nodes_neighbor_times", merged_padded_nodes_neighbor_times)
        # print("merged_padded_nodes_neighbor_ids", merged_padded_nodes_neighbor_ids)
        # print("merged_padded_nodes_edge_ids", merged_padded_nodes_edge_ids)
        # print("merged_padded_nodes_neighbor_co_occurrence_features", merged_padded_nodes_neighbor_co_occurrence_features[:,:,0])
        # print("merged_src_node_indices", merged_src_node_indices)
        # print("merged_dst_node_indices", merged_dst_node_indices)

        return (
            merged_padded_nodes_neighbor_ids,
            merged_padded_nodes_edge_ids,
            merged_padded_nodes_neighbor_times,
            merged_padded_nodes_neighbor_co_occurrence_features,
            merged_src_node_indices,
            merged_dst_node_indices,
        )

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the
        results for uniform and time_interval_aware sampling) :param neighbor_sampler:

        NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class TransformerDecoder(nn.Module):
    """DyGDecoder's transformer decoder."""

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super().__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
                nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(attention_dim), nn.LayerNorm(attention_dim)]
        )

    def forward(self, inputs: torch.Tensor):
        """Process the inputs by Transformer decoder :param inputs: Tensor, shape (batch_size,
        num_patches, self.attention_dim) :return:"""
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        # print(f"transposed_inputs has nan? {torch.isnan(transposed_inputs).any()}")
        # print(transposed_inputs)
        transposed_inputs_post_norm = self.norm_layers[0](transposed_inputs)
        # print(f"transposed_inputs after norm has nan? {torch.isnan(transposed_inputs_post_norm).any()}")
        # if torch.isnan(transposed_inputs_post_norm).any():
        #     print(f"before transpose: {transposed_inputs}")
        #     print(f"after transpose: {transposed_inputs_post_norm}")
        # print(transposed_inputs)
        # we create a mask to make the query token attend to only the previous tokens
        mask = torch.triu(
            torch.full(
                (transposed_inputs_post_norm.size(0), transposed_inputs_post_norm.size(0)),
                float("-inf"),
                dtype=torch.float32,
                device=transposed_inputs_post_norm.device,
            ),
            diagonal=1,
        )
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs_post_norm,
            key=transposed_inputs_post_norm,
            value=transposed_inputs_post_norm,
            attn_mask=mask,
        )[0].transpose(0, 1)
        # print(f"hidden_states has nan? {torch.isnan(hidden_states).any()}")
        # hidden_states, attn_weights = self.multi_head_attention(
        #     query=transposed_inputs, key=transposed_inputs, value=transposed_inputs, attn_mask=mask
        # )
        # hidden_states = hidden_states.transpose(0, 1)

        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs


class SelfCrossAttention(nn.Module):
    """Self Cross Attention module."""

    def __init__(self, d, H, T, dropout: float = 0.1):
        """Initialize the SelfCrossAttention module.

        :param d: int, size of feature dimension
        :param H: int, number of attention heads
        :param T: int, maximum length of input sequences (in tokens)
        :param dropout: float, dropout rate
        """
        super().__init__()
        assert d % H == 0
        self.d = d
        self.H = H
        self.T = T
        self.dropout = dropout

        # self and cross attention parameters
        # key, query, value projections for all heads, but in a batch
        # output is 3x the dimension because it includes key, query, and value
        self.self_attn = nn.Linear(d, 3 * d)
        self.cross_attn = nn.Linear(d, 3 * d)

        self.proj = nn.Linear(d, d)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        self_seq: torch.Tensor,
        cross_seq: torch.Tensor,
        self_seq_time: torch.Tensor,
        cross_seq_time: torch.Tensor,
    ):
        """
        :param self_seq: Tensor, shape (batch size, sequence length, d)
        :param cross_seq: Tensor, shape (batch size, sequence length, d)
        :param self_seq_time: Tensor, shape (batch size, sequence length)
        :param cross_seq_time: Tensor, shape (batch size, sequence length)
        """
        B_self, T_self, _ = self_seq.size()  # batch size, self_seq length, _
        B_cross, T_cross, _ = cross_seq.size()  # batch size, cross_seq length, _
        assert B_self == B_cross

        q_self, k_self, v_self = self.self_attn(self_seq).split(
            self.d, dim=2
        )  # (batch size, self_seq length, d)
        k_self = k_self.view(B_self, T_self, self.H, self.d // self.H).transpose(
            1, 2
        )  # (batch size, H, self_seq length, d // H)
        q_self = q_self.view(B_self, T_self, self.H, self.d // self.H).transpose(1, 2)
        v_self = v_self.view(B_self, T_self, self.H, self.d // self.H).transpose(1, 2)

        q_cross, k_cross, v_cross = self.cross_attn(cross_seq).split(
            self.d, dim=2
        )  # (batch size, cross_seq length, d)
        k_cross = k_cross.view(B_cross, T_cross, self.H, self.d // self.H).transpose(
            1, 2
        )  # (batch size, H, cross_seq length, d // H)
        q_cross = q_cross.view(B_cross, T_cross, self.H, self.d // self.H).transpose(1, 2)
        v_cross = v_cross.view(B_cross, T_cross, self.H, self.d // self.H).transpose(1, 2)

        q = torch.cat(
            [q_self, q_cross], dim=2
        )  # (batch size, H, self_seq length + cross_seq length, d // H)
        k = torch.cat([k_self, k_cross], dim=2)
        v = torch.cat([v_self, v_cross], dim=2)

        combined_seq_time = torch.cat([self_seq_time, cross_seq_time], dim=1)
        # print("self_seq_time", self_seq_time)
        # print("cross_seq_time", cross_seq_time)
        mask = (
            (combined_seq_time.unsqueeze(2) >= combined_seq_time.unsqueeze(1))
            .bool()
            .unsqueeze(1)
            .expand(-1, self.H, -1, -1)
            .to(self_seq.device)
        )  # (batch size, H, self_seq length + cross_seq length, self_seq length + cross_seq length)
        # print("mask", mask)
        # print("q shape", q.shape)
        # print("k shape", k.shape)
        # print("v shape", v.shape)
        # print("mask shape", mask.shape)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0
            )
        else:
            att = (
                q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            )  # (batch size, H, self_seq length + cross_seq length, self_seq length + cross_seq length)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            y = att @ v  # (batch size, H, self_seq length + cross_seq length, d // H)

        y = y.transpose(1, 2).contiguous().view(B_self, T_self + T_cross, self.d)
        y = self.resid_dropout(self.proj(y))

        return y[:, :T_self], y[:, T_self:]  # split to self and cross parts then return


class SelfCrossFormer(nn.Module):
    """SelfCrossFormer model."""

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_cross_attn = SelfCrossAttention(attention_dim, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
                nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(attention_dim), nn.LayerNorm(attention_dim)]
        )

    def forward(
        self,
        self_seq: torch.Tensor,
        cross_seq: torch.Tensor,
        self_seq_time: torch.Tensor,
        cross_seq_time: torch.Tensor,
    ):
        """Process two sequences by SelfCrossFormer :param self_seq: Tensor, shape (batch_size,
        num_self_patches, self.attention_dim) :param cross_seq: Tensor, shape (batch_size,
        num_cross_patches, self.attention_dim) :param self_seq_time: Tensor, shape (batch_size,
        num_self_patches) :param cross_seq_time: Tensor, shape (batch_size, num_cross_patches)
        :return:"""

        self_seq_len, cross_seq_len = self_seq.size(1), cross_seq.size(1)
        inputs = torch.cat(
            [self_seq, cross_seq], dim=1
        )  # (batch size, num_self_patches + num_cross_patches, self.attention_dim)
        normalized_inputs = self.norm_layers[0](inputs)

        # Tensor, shape (batch_size, num_self_patches, self.attention_dim)
        # Tensor, shape (batch_size, num_cross_patches, self.attention_dim)
        self_hidden_states, cross_hidden_states = self.self_cross_attn(
            self_seq=normalized_inputs[:, :self_seq_len],
            cross_seq=normalized_inputs[:, self_seq_len:],
            self_seq_time=self_seq_time,
            cross_seq_time=cross_seq_time,
        )

        # Tensor, shape (batch_size, num_self_patches + num_cross_patches, self.attention_dim)
        outputs = inputs + self.dropout(
            torch.cat([self_hidden_states, cross_hidden_states], dim=1)
        )
        # Tensor, shape (batch_size, num_self_patches + num_cross_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        # Tensor, shape (batch_size, num_self_patches + num_cross_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return (
            outputs[:, :self_seq_len],
            outputs[:, self_seq_len:],
        )  # split to self and cross parts then return
