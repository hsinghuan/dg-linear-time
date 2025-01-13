import argparse
import math
import os
import pickle  # nosec

# nosec comment needed to pass the bandit test
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

sys.path.append("../")
from src.models.modules.dygdecoder import TransformerDecoder
from src.models.modules.dygformer import TransformerEncoder
from src.models.modules.time import (
    CosineTimeEncoder,
    LinearTimeEncoder,
    SineCosineTimeEncoder,
)


class SyntheticDataset(Dataset):
    """Synthetic Dataset Class."""

    def __init__(
        self,
        N=100,  # number of sequences
        M=7,  # max sequence length
        w=0.1,  # decay weight
        lam=0.01,  # rate for Exp(\lambda)
        seed=42,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.w = w
        self.lam = lam
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Pre-generate all sequences
        self.data = []
        for _ in range(N):
            # 1) Generate times t_1, ..., t_M using exponential interarrival
            ts = []
            current_time = 0.0
            for _m in range(M):
                delta = np.random.exponential(1.0 / lam)
                current_time += delta
                ts.append(current_time)

            # 2) Generate x_m in {+1, -1} w.p. 0.5 each
            xs = [1 if random.random() < 0.5 else -1 for _m in range(M)]  # nosec
            # nosec comment needed to pass the bandit test

            # 3) Define target time t (e.g., t_M + Exp(1/lambda))
            t_target = current_time + np.random.exponential(1.0 / lam)

            # 4) Compute label y
            weighted_sum = 0.0
            for i, (tm, xm) in enumerate(zip(ts, xs)):
                weighted_sum += math.exp(-w * (t_target - tm)) * xm
                # print("i:", i, "tm:", tm, "weights:", math.exp(-w * (t_target - tm)))
            weighted_sum += np.random.normal(scale=0.1)  # add Gaussian noise
            y = 1 if weighted_sum > 0 else 0

            # Directly provide time difference features
            # Put the target event at the end of the sequence
            t_diff = [t_target - t for t in ts]
            t_diff.append(0.0)
            t_diff = torch.tensor(t_diff).float()
            xs.append(0)
            xs = torch.tensor(xs).float()
            self.data.append((xs, t_diff, y))

    def __len__(self):
        """Return the number of sequences."""
        return self.N

    def __getitem__(self, idx):
        """Return a sequence."""
        return self.data[idx]


class DecayOscillateDataset(Dataset):
    """Synthetic Decay + Oscillate Dataset Class."""

    def __init__(
        self,
        N=100,  # number of sequences
        M=7,  # max sequence length
        w=0.003,  # decay weight
        omega=0.02,  # frequency of oscillation
        lam=0.01,  # rate for Exp(\lambda)
        seed=42,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.w = w
        self.lam = lam
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Pre-generate all sequences
        self.data = []
        for _ in range(N):
            # 1) Generate times t_1, ..., t_M using exponential interarrival
            ts = []
            current_time = 0.0
            for _m in range(M):
                delta = np.random.exponential(1.0 / lam)
                current_time += delta
                ts.append(current_time)

            # 2) Generate x_m in {+1, -1} w.p. 0.5 each
            xs = [1 if random.random() < 0.5 else -1 for _m in range(M)]  # nosec
            # nosec comment needed to pass the bandit test

            # 3) Define target time t (e.g., t_M + Exp(1/lambda))
            t_target = current_time + np.random.exponential(1.0 / lam)

            # 4) Compute label y
            weighted_sum = 0.0
            for i, (tm, xm) in enumerate(zip(ts, xs)):
                weighted_sum += (
                    np.exp(-w * (t_target - tm)) * np.cos(omega * (t_target - tm)) ** 2
                ) * xm
            # print([t_target - tm for tm in ts])
            # print(
            #     [
            #         (np.exp(-w * (t_target - tm)) * np.cos(omega * (t_target - tm)) ** 2)
            #         for tm in ts
            #     ]
            # )

            weighted_sum += np.random.normal(scale=0.1)  # add Gaussian noise
            y = 1 if weighted_sum > 0 else 0

            # Directly provide time difference features
            # Put the target event at the end of the sequence
            t_diff = [t_target - t for t in ts]
            t_diff.append(0.0)
            t_diff = torch.tensor(t_diff).float()
            xs.append(0)
            xs = torch.tensor(xs).float()
            self.data.append((xs, t_diff, y))

    def __len__(self):
        """Return the number of sequences."""
        return self.N

    def __getitem__(self, idx):
        """Return a sequence."""
        return self.data[idx]


class SeqClsModel(nn.Module):
    def __init__(
        self,
        time_encoding_method,
        time_feat_dim,
        attn_dim,
        num_layers=1,
        num_heads=1,
        autoregressive=False,
        add_bos=False,
        avg_time_diff=0.0,
        std_time_diff=1.0,
        get_attn_score=False,
    ):
        super().__init__()

        if time_encoding_method == "linear":
            self.time_encoder = LinearTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "sinusoidal":
            self.time_encoder = CosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        elif time_encoding_method == "sinecosine":
            self.time_encoder = SineCosineTimeEncoder(
                time_dim=time_feat_dim, mean=avg_time_diff, std=std_time_diff
            )
        else:
            raise ValueError(f"Invalid time_encoding_method: {time_encoding_method}")

        self.add_bos = add_bos
        if autoregressive:
            self.transformers = nn.ModuleList(
                [
                    TransformerDecoder(
                        attention_dim=attn_dim,
                        num_heads=num_heads,
                        dropout=0.1,
                    )
                    for _ in range(num_layers)
                ]
            )
            if self.add_bos:
                self.bos_embedding = nn.Parameter(
                    torch.empty(1, attn_dim), requires_grad=True
                )
                nn.init.normal_(self.bos_embedding)
        else:
            self.transformers = nn.ModuleList(
                [
                    TransformerEncoder(
                        attention_dim=attn_dim,
                        num_heads=num_heads,
                        dropout=0.1,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.classifier = nn.Linear(attn_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.autoregressive = autoregressive
        self.get_attn_score = get_attn_score

    def forward(self, times, features):
        """Predict the probability of the target being 1."""
        batch_size = times.size(0)
        time_feat = self.time_encoder(times)  # (batch_size, seq_len, time_feat_dim)
        features = features.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = torch.cat([time_feat, features], dim=-1)  # (batch_size, seq_len, time_feat_dim + 1)
        if self.autoregressive and self.add_bos:
            bos = self.bos_embedding.view(1, 1, -1).repeat(batch_size, 1, 1)
            x = torch.cat([bos, x], dim=1)
        for transformer in self.transformers:
            x = transformer(x, get_attn_score=self.get_attn_score)
            if self.get_attn_score:
                x, attn_score = x
        if self.autoregressive:
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        x = self.classifier(x)
        if self.get_attn_score:
            return self.sigmoid(x), attn_score
        else:
            return self.sigmoid(x)


def train_loop(model, optimizer, criterion, dataloader, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_num = 0
    for x, t_diff, y in dataloader:
        x = x.to(device)
        t_diff = t_diff.to(device)
        y = y.to(device)
        y_pred = model(t_diff, x)
        if model.get_attn_score:
            y_pred, _ = y_pred
        loss = criterion(y_pred, y.float().unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_num += batch_size
    loss = total_loss / total_num
    return loss


@torch.no_grad()
def eval_loop(model, criterion, dataloader, device):
    """Evaluate the model."""
    assert model.get_attn_score is False
    model.eval()
    y_pred_list = []
    y_list = []
    for x, t_diff, y in dataloader:
        x = x.to(device)
        t_diff = t_diff.to(device)
        y = y.to(device)
        y_pred = model(t_diff, x)
        y_pred_list.append(y_pred)
        y_list.append(y)
    y_pred = torch.cat(y_pred_list, dim=0)
    y = torch.cat(y_list, dim=0)
    loss = criterion(y_pred, y.float().unsqueeze(-1))
    acc = accuracy_score(y.cpu().numpy(), (y_pred.cpu().numpy() > 0.5).astype(int))
    return loss, acc


@torch.no_grad()
def get_attn_score(model, dataloader, device):
    """Get attention scores."""
    assert model.get_attn_score is True
    model.eval()
    attn_score_list = []
    t_diff_list = []
    for x, t_diff, y in dataloader:
        x = x.to(device)
        t_diff = t_diff.to(device)
        y = y.to(device)
        _, attn_score = model(t_diff, x)
        attn_score_list.append(attn_score)
        t_diff_list.append(t_diff)
    attn_score = torch.cat(attn_score_list, dim=0)
    t_diff = torch.cat(t_diff_list, dim=0)
    return attn_score, t_diff


def oracle_loop(w, dataloader):
    """Predict with the oracle model."""
    y_pred_list = []
    y_list = []
    for x, t_diff, y in dataloader:
        y_pred = torch.sum(torch.exp(-w * t_diff) * x, dim=1) > 0
        y_pred_list.append(y_pred)
        y_list.append(y)
    y_pred = torch.cat(y_pred_list, dim=0)
    y = torch.cat(y_list, dim=0)
    acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
    return acc


def get_mean_std(loader):
    """Get the mean and std of time difference features."""
    mean_time_diff_per_seq = []
    for _, t_diff, _ in loader:
        mean = t_diff.mean()
        mean_time_diff_per_seq.append(mean)
    mean = np.mean([mean_time_diff_per_seq])
    std = np.std([mean_time_diff_per_seq])
    return mean, std


def get_device(device):
    """Get the device."""
    if device is None:
        device = "cpu"
    else:
        device = f"cuda:{device}"
    return device


def experiment(args):
    """A single run of the experiment."""
    # create dataset and dataloaders
    device = get_device(args.device)
    if args.dataset == "synthetic":
        dataset = SyntheticDataset(N=args.N, M=args.M, w=args.w, lam=args.lam, seed=args.seed)
    elif args.dataset == "decay_oscillate":
        dataset = DecayOscillateDataset(N=args.N, M=args.M, w=args.w, omega=args.omega, lam=args.lam, seed=args.seed)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # oracle
    if args.oracle:
        return (
            oracle_loop(args.w, train_loader),
            oracle_loop(args.w, val_loader),
            oracle_loop(args.w, test_loader),
        )

    # create model, optimizer, and loss function
    if args.scale:
        avg_time_diff, std_time_diff = get_mean_std(train_loader)
    else:
        avg_time_diff, std_time_diff = 0, 1

    model = SeqClsModel(
        time_encoding_method=args.time_encoding_method,
        time_feat_dim=args.time_feat_dim,
        attn_dim=args.time_feat_dim + 1,
        num_layers=args.num_layers,
        num_heads=1,
        autoregressive=args.autoregressive,
        add_bos=args.add_bos,
        avg_time_diff=avg_time_diff,
        std_time_diff=std_time_diff,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # train
    staleness = 0
    best_model = None
    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        train_loss = train_loop(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = eval_loop(model, criterion, val_loader, device)
        print(
            f"Seed: {args.seed} Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)
            staleness = 0
        else:
            staleness += 1
            if staleness >= args.patience:
                break

    model = best_model
    train_loss, train_acc = eval_loop(model, criterion, train_loader, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    val_loss, val_acc = eval_loop(model, criterion, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    test_loss, test_acc = eval_loop(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if args.get_attn_score:
        model.get_attn_score = True
        train_attn_score, train_t_diff = get_attn_score(model, train_loader, device)
        val_attn_score, val_t_diff = get_attn_score(model, val_loader, device)
        test_attn_score, test_t_diff = get_attn_score(model, test_loader, device)
        return (
            train_acc,
            val_acc,
            test_acc,
            train_attn_score,
            val_attn_score,
            test_attn_score,
            train_t_diff,
            val_t_diff,
            test_t_diff,
        )
    else:
        return train_acc, val_acc, test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--N", type=int, default=2000, help="number of sequences")
    argparser.add_argument(
        "--M", type=int, default=7, help="sequence length (not including target)"
    )
    argparser.add_argument("--w", type=float, default=0.003, help="exponential decay weight")
    argparser.add_argument("--lam", type=float, default=0.01, help=r"rate for Exp(\lambda)")
    argparser.add_argument("--omega", type=float, default=0.02, help="frequency of oscillation")
    argparser.add_argument("--batch_size", type=int, default=32, help="batch size")
    argparser.add_argument(
        "--time_encoding_method", type=str, default="sinusoidal", help="time encoding method"
    )
    argparser.add_argument(
        "--dataset", type=str, default="decay_oscillate", help="which dataset to use, synthetic or decay_oscillate"
    )
    argparser.add_argument("--time_feat_dim", type=int, default=2, help="time feature dimension")
    argparser.add_argument(
        "--num_layers", type=int, default=1, help="number of transformer layers"
    )
    argparser.add_argument(
        "--autoregressive", action="store_true", help="use autoregressive attention"
    )
    argparser.add_argument(
        "--add_bos", action="store_true", help="add BOS token if using autoregressive attention"
    )
    argparser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    argparser.add_argument("--scale", action="store_true", help="scale time difference features")
    argparser.add_argument("--runs", type=int, default=10, help="number of runs")
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument(
        "--device", type=int, default=None, help="device number, use CPU if None"
    )
    argparser.add_argument("--oracle", action="store_true", help="use oracle")
    argparser.add_argument("--get_attn_score", action="store_true", help="get attention scores")
    args = argparser.parse_args()

    def multiple_seeds_run(args):
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        for seed in range(args.runs):
            args.seed = seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            train_acc, val_acc, test_acc = experiment(args)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
        return train_acc_list, val_acc_list, test_acc_list

    if args.get_attn_score:
        output_dir = "attn_analysis_output"
        os.makedirs("attn_analysis_output", exist_ok=True)
        assert (
            args.autoregressive is True
        )  # only do attn score computation for autoregressive model
        args.seed = 42
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        (
            train_acc,
            val_acc,
            test_acc,
            train_attn_score,
            val_attn_score,
            test_attn_score,
            train_t_diff,
            val_t_diff,
            test_t_diff,
        ) = experiment(args)
        attn_score_dict = {
            "train_attn_score": train_attn_score,
            "train_t_diff": train_t_diff,
            "val_attn_score": val_attn_score,
            "val_t_diff": val_t_diff,
            "test_attn_score": test_attn_score,
            "test_t_diff": test_t_diff,
        }
        torch.save(
            attn_score_dict,
            os.path.join(
                output_dir,
                f"w_{args.w}_num_layers_{args.num_layers}_time_feat_dim_{args.time_feat_dim}_{args.time_encoding_method}_attn_score.pt",
            ),
        )
    else:
        output_dir = "synthetic_output"
        os.makedirs("synthetic_output", exist_ok=True)
        performance_dict = {}
        args.oracle = False
        for num_layers in [1, 2]:
            for time_feat_dim in [2, 4, 8, 16]:
                for time_encoding_method in ["sinusoidal", "sinecosine", "linear"]:
                    if time_encoding_method == "linear":
                        args.scale = True
                    else:
                        args.scale = False

                    for autoregressive in [True, False]:
                        args.num_layers = num_layers
                        args.time_feat_dim = time_feat_dim
                        args.time_encoding_method = time_encoding_method
                        args.autoregressive = autoregressive
                        if autoregressive:
                            for add_bos in [True, False]:
                                args.add_bos = add_bos
                                train_acc_list, val_acc_list, test_acc_list = multiple_seeds_run(
                                    args
                                )
                                performance_dict[
                                    (
                                        f"num_layers:{num_layers}",
                                        f"time_feat_dim:{time_feat_dim}",
                                        f"time_encoding_method:{time_encoding_method}",
                                        f"autoregressive:{autoregressive}",
                                        f"add_bos:{add_bos}",
                                    )
                                ] = [
                                    (np.mean(train_acc_list), np.std(train_acc_list)),
                                    (np.mean(val_acc_list), np.std(val_acc_list)),
                                    (np.mean(test_acc_list), np.std(test_acc_list)),
                                ]
                        else:
                            args.add_bos = False
                            train_acc_list, val_acc_list, test_acc_list = multiple_seeds_run(args)
                            performance_dict[
                                (
                                    f"num_layers:{num_layers}",
                                    f"time_feat_dim:{time_feat_dim}",
                                    f"time_encoding_method:{time_encoding_method}",
                                    f"autoregressive:{autoregressive}",
                                )
                            ] = [
                                (np.mean(train_acc_list), np.std(train_acc_list)),
                                (np.mean(val_acc_list), np.std(val_acc_list)),
                                (np.mean(test_acc_list), np.std(test_acc_list)),
                            ]

        args.oracle = True
        oracle_train_acc_list, oracle_val_acc_list, oracle_test_acc_list = multiple_seeds_run(args)
        performance_dict["oracle"] = [
            (np.mean(oracle_train_acc_list), np.std(oracle_train_acc_list)),
            (np.mean(oracle_val_acc_list), np.std(oracle_val_acc_list)),
            (np.mean(oracle_test_acc_list), np.std(oracle_test_acc_list)),
        ]

        with open(os.path.join(output_dir, f"{args.w}_performance_dict.pkl"), "wb") as file:
            pickle.dump(performance_dict, file)

        # with open("synthetic.txt", "w") as file:
        #     for key, value in performance_dict.items():
        #         if isinstance(key, tuple):
        #             file.write(f"{key} Train Acc: {round(value[0][0], 4)} ± {round(value[0][1], 4)} Val Acc: {round(value[1][0], 4)} ± {round(value[1][1], 4)} Test Acc: {round(value[2][0], 4)} ± {round(value[2][1], 4)}\n")
        #         else:
        #             file.write(f"{key} Train Acc: {round(value[0][0], 4)} ± {round(value[0][1], 4)} Val Acc: {round(value[1][0], 4)} ± {round(value[1][1], 4)} Test Acc: {round(value[2][0], 4)} ± {round(value[2][1], 4)}\n")
