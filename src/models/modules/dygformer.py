# Copied from https://github.com/yule-BUAA/DyGLib_TGB/blob/master/models/DyGFormer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from src.models.modules.time import (
    CosineTimeEncoder,
    NoTimeEncoder,
    SineCosineTimeEncoder,
)
from src.utils.analysis import analyze_target_historical_event_time_diff
from src.utils.data import NeighborSampler


class DyGFormer(nn.Module):
    """DyGFormer model."""

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
        embed_method: str = "original",  # original dygformer merges the two sequences when forwarding through transformer
        time_encoding_method: str = "sinusoidal",
        avg_time_diff: float = None,
        std_time_diff: float = None,
        time_channel_embedding_dim: int = None,
        use_positional_embedding: bool = False,
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
        self.embed_method = embed_method
        self.device = device
        self.use_positional_embedding = use_positional_embedding

        if use_positional_embedding:
            assert (
                avg_time_diff == 0 and std_time_diff == 1 and time_encoding_method == "sinusoidal"
            )

        if time_encoding_method == "sinusoidal":
            self.time_encoder = CosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "sinecosine":
            self.time_encoder = SineCosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "linear":
            self.time_encoder = NoTimeEncoder(mean=avg_time_diff, std=std_time_diff)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
        )

        if time_channel_embedding_dim is not None:
            self.time_channel_embedding_dim = time_channel_embedding_dim
        else:
            self.time_channel_embedding_dim = channel_embedding_dim

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
                "time": nn.Linear(
                    in_features=self.patch_size * self.time_feat_dim,
                    out_features=self.time_channel_embedding_dim,
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
        attention_dim = (
            self.num_channels - 1
        ) * self.channel_embedding_dim + self.time_channel_embedding_dim

        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    attention_dim=attention_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_layer = nn.Linear(
            in_features=attention_dim,
            out_features=self.output_dim,
            bias=True,
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        analyze_length: bool = False,
        analyze_attn_scores: bool = False,
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

        if analyze_attn_scores:
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
        else:
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

        if self.embed_method == "original":
            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
            patches_nodes_neighbor_node_raw_features = torch.cat(
                [
                    src_patches_nodes_neighbor_node_raw_features,
                    dst_patches_nodes_neighbor_node_raw_features,
                ],
                dim=1,
            )
            patches_nodes_edge_raw_features = torch.cat(
                [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1
            )
            patches_nodes_neighbor_time_features = torch.cat(
                [
                    src_patches_nodes_neighbor_time_features,
                    dst_patches_nodes_neighbor_time_features,
                ],
                dim=1,
            )
            patches_nodes_neighbor_co_occurrence_features = torch.cat(
                [
                    src_patches_nodes_neighbor_co_occurrence_features,
                    dst_patches_nodes_neighbor_co_occurrence_features,
                ],
                dim=1,
            )

            if self.time_channel_embedding_dim == self.channel_embedding_dim:
                # regular forward implementation
                patches_data = [
                    patches_nodes_neighbor_node_raw_features,
                    patches_nodes_edge_raw_features,
                    patches_nodes_neighbor_time_features,
                    patches_nodes_neighbor_co_occurrence_features,
                ]
                # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
                patches_data = torch.stack(patches_data, dim=2)
                # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
                patches_data = patches_data.reshape(
                    batch_size,
                    src_num_patches + dst_num_patches,
                    self.num_channels * self.channel_embedding_dim,
                )
            else:
                # special implementation to adapt to different time channel embedding dim
                patches_nodes_neighbor_node_raw_features.reshape(
                    batch_size, src_num_patches + dst_num_patches, self.channel_embedding_dim
                )
                patches_nodes_edge_raw_features.reshape(
                    batch_size, src_num_patches + dst_num_patches, self.channel_embedding_dim
                )
                patches_nodes_neighbor_time_features.reshape(
                    batch_size, src_num_patches + dst_num_patches, self.time_channel_embedding_dim
                )
                patches_nodes_neighbor_co_occurrence_features.reshape(
                    batch_size, src_num_patches + dst_num_patches, self.channel_embedding_dim
                )
                patches_data = torch.cat(
                    [
                        patches_nodes_neighbor_node_raw_features,
                        patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features,
                        patches_nodes_neighbor_co_occurrence_features,
                    ],
                    dim=-1,
                )  # (batch_size, src_num_patches + dst_num_patches, (num_channels - 1) * channel_embedding_dim + time_channel_embedding_dim)

            # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
            for transformer in self.transformers:
                patches_data = transformer(patches_data)

            # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            src_patches_data = patches_data[:, :src_num_patches, :]
            # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
            dst_patches_data = patches_data[
                :, src_num_patches : src_num_patches + dst_num_patches, :
            ]
            # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            src_patches_data = torch.mean(src_patches_data, dim=1)
            # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            dst_patches_data = torch.mean(dst_patches_data, dim=1)

            # Tensor, shape (batch_size, output_dim)
            src_node_embeddings = self.output_layer(src_patches_data)
            # Tensor, shape (batch_size, output_dim)
            dst_node_embeddings = self.output_layer(dst_patches_data)

        elif self.embed_method == "separate":
            src_patches_data = [
                src_patches_nodes_neighbor_node_raw_features,
                src_patches_nodes_edge_raw_features,
                src_patches_nodes_neighbor_time_features,
                src_patches_nodes_neighbor_co_occurrence_features,
            ]
            # Tensor, shape (batch_size, src_num_patches, num_channels, channel_embedding_dim)
            src_patches_data = torch.stack(src_patches_data, dim=2)
            # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            src_patches_data = src_patches_data.reshape(
                batch_size,
                src_num_patches,
                self.num_channels * self.channel_embedding_dim,
            )

            dst_patches_data = [
                dst_patches_nodes_neighbor_node_raw_features,
                dst_patches_nodes_edge_raw_features,
                dst_patches_nodes_neighbor_time_features,
                dst_patches_nodes_neighbor_co_occurrence_features,
            ]
            # Tensor, shape (batch_size, dst_num_patches, num_channels, channel_embedding_dim)
            dst_patches_data = torch.stack(dst_patches_data, dim=2)
            # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
            dst_patches_data = dst_patches_data.reshape(
                batch_size,
                dst_num_patches,
                self.num_channels * self.channel_embedding_dim,
            )
            if analyze_attn_scores:
                for transformer in self.transformers:
                    src_patches_data, src_attn_scores = transformer(
                        src_patches_data, get_attn_score=True
                    )
                    dst_patches_data, dst_attn_scores = transformer(
                        dst_patches_data, get_attn_score=True
                    )
                src_attn_scores_analysis = {
                    "t": src_patches_nodes_neighbor_times,
                    "attn_scores": src_attn_scores,
                }
                dst_attn_scores_analysis = {
                    "t": dst_patches_nodes_neighbor_times,
                    "attn_scores": dst_attn_scores,
                }
            else:
                # Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
                for transformer in self.transformers:
                    src_patches_data = transformer(src_patches_data)
                # Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
                for transformer in self.transformers:
                    dst_patches_data = transformer(dst_patches_data)

            # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            src_patches_data = torch.mean(src_patches_data, dim=1)
            # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
            dst_patches_data = torch.mean(dst_patches_data, dim=1)

            # Tensor, shape (batch_size, output_dim)
            src_node_embeddings = self.output_layer(src_patches_data)
            # Tensor, shape (batch_size, output_dim)
            dst_node_embeddings = self.output_layer(dst_patches_data)

        if analyze_length and analyze_attn_scores:
            return (
                src_node_embeddings,
                dst_node_embeddings,
                src_history_length_analysis,
                dst_history_length_analysis,
                src_attn_scores_analysis,
                dst_attn_scores_analysis,
            )
        elif analyze_length:
            return (
                src_node_embeddings,
                dst_node_embeddings,
                src_history_length_analysis,
                dst_history_length_analysis,
            )
        elif analyze_attn_scores:
            return (
                src_node_embeddings,
                dst_node_embeddings,
                src_attn_scores_analysis,
                dst_attn_scores_analysis,
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
        """Pad the sequences for nodes in node_ids :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ) :param nodes_neighbor_ids_list:

        list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids :param
        nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor
        interaction timestamp for nodes in node_ids :param patch_size: int, patch size :param
        max_input_sequence_length: int, maximal number of neighbors for each node :return:
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
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[
                    idx, 1 : len(nodes_neighbor_ids_list[idx]) + 1
                ] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[
                    idx, 1 : len(nodes_edge_ids_list[idx]) + 1
                ] = nodes_edge_ids_list[idx]
                if self.use_positional_embedding:
                    pos_list = np.flip(
                        np.unique(nodes_neighbor_times_list[idx])
                    )  # newer timestamp will have lower position index

                    time_to_pos = {t: i for i, t in enumerate(pos_list)}
                    padded_nodes_neighbor_times[idx, 0] = 0
                    if (
                        nodes_neighbor_times_list[idx][-1] == node_interact_times[idx]
                    ):  # if target time is the same as the last one in sequence, offset is 0
                        offset = 0
                    else:  # otherwise, add 1 to the offset
                        offset = 1

                    padded_nodes_neighbor_times[
                        idx, 1 : len(nodes_neighbor_times_list[idx]) + 1
                    ] = np.array(
                        [time_to_pos[t] + offset for t in nodes_neighbor_times_list[idx]]
                    )  # just use the relative position index as the timestamp
                else:
                    padded_nodes_neighbor_times[
                        idx, 1 : len(nodes_neighbor_times_list[idx]) + 1
                    ] = nodes_neighbor_times_list[idx]
            else:
                if (
                    self.use_positional_embedding
                ):  # convert the timestamp to the relative position index
                    padded_nodes_neighbor_times[idx, 0] = 0

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
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        if self.use_positional_embedding:
            padded_nodes_neighbor_time_features = time_encoder(
                timestamps=torch.from_numpy(
                    padded_nodes_neighbor_times
                )  # simply encode the index as the timestamp
                .float()
                .to(self.device)
            )
        else:
            padded_nodes_neighbor_time_features = time_encoder(
                timestamps=torch.from_numpy(
                    node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
                )
                .float()
                .to(self.device)
            )
        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        # print("padded nodes neighbor time features shape", padded_nodes_neighbor_time_features.shape)
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
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
        patch_size: int = 1,
        padded_nodes_neighbor_times: torch.Tensor = None,
    ):
        """Get the sequence of patches for nodes :param padded_nodes_neighbor_node_raw_features:

        Tensor, shape (batch_size, max_seq_length, node_feat_dim) :param
        padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length,
        time_feat_dim) :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape
        (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim) :param patch_size: int, patch
        size :return:
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

        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        if padded_nodes_neighbor_times is not None:
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

        return (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
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


class NeighborCooccurrenceEncoder(nn.Module):
    """Neighbor co-occurrence encoder."""

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = "cpu"):
        """
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super().__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

    def count_nodes_appearances(
        self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray
    ):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(
            src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids
        ):
            # RUNNING EXAMPLE: src_padded_node_neighbor_ids = [1,2,1,0], dst_padded_node_neighbor_ids = [2,2,1,3]
            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = (
                torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            )
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            # RUNNING EXAMPLE: src_padded_node_neighbor_counts_in_src = [2,1,2,1], src_mapping_dict = {0: 1, 1: 2, 2: 1}

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True
            )
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = (
                torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            )
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))
            # RUNNING EXAMPLE: dst_padded_node_neighbor_counts_in_dst = [2,2,1,1], dst_mapping_dict = {0: 0, 1: 1, 2: 2, 3: 1}

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = (
                torch.from_numpy(src_padded_node_neighbor_ids.copy())
                .apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0))
                .float()
                .to(self.device)
            )
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack(
                    [
                        src_padded_node_neighbor_counts_in_src,
                        src_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )
            # RUNNING EXAMPLE: src_padded_node_neighbor_counts_in_dst = [1, 2, 1, 0]
            # RUNNING EXAMPLE: src_padded_nodes_appearances = [[[2, 1], [1, 2], [2, 1], [1, 0]]]

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = (
                torch.from_numpy(dst_padded_node_neighbor_ids.copy())
                .apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0))
                .float()
                .to(self.device)
            )
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack(
                    [
                        dst_padded_node_neighbor_counts_in_src,
                        dst_padded_node_neighbor_counts_in_dst,
                    ],
                    dim=1,
                )
            )
            # RUNNING EXAMPLE: dst_padded_node_neighbor_counts_in_src = [1, 1, 2, 0]
            # RUNNING EXAMPLE: dst_padded_nodes_appearances = [[[1, 2], [1, 2], [2, 1], [0, 1]]]

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(
        self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray
    ):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        )

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = (
            self.neighbor_co_occurrence_encode_layer(
                src_padded_nodes_appearances.unsqueeze(dim=-1)
            ).sum(dim=2)
        )
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = (
            self.neighbor_co_occurrence_encode_layer(
                dst_padded_nodes_appearances.unsqueeze(dim=-1)
            ).sum(dim=2)
        )

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        )


class TransformerEncoder(nn.Module):
    """DyGFormer's transformer encoder."""

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

    def forward(self, inputs: torch.Tensor, get_attn_score: bool = False):
        """Encode the inputs by Transformer encoder :param inputs: Tensor, shape (batch_size,
        num_patches, self.attention_dim) :return:"""
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states, attn_scores = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs
        )
        hidden_states = hidden_states.transpose(0, 1)

        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        if get_attn_score:
            return outputs, attn_scores
        else:
            return outputs
