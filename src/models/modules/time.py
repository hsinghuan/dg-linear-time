import numpy as np
import torch
import torch.nn as nn


class CosineTimeEncoder(nn.Module):
    """Time encoder for encoding the elapsed time into a ``time_dim'' dimensional vector."""

    def __init__(
        self,
        time_dim: int,
        parameter_requires_grad: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        """
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(
                time_dim, -1
            )
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False
        self.mean = mean
        self.std = std

    def forward(self, timestamps: torch.Tensor):
        """Compute time encodings of time in timestamps :param timestamps: Tensor, shape
        (batch_size, seq_len) :return:"""
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
        timestamps = (timestamps - self.mean) / self.std
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))
        # print("output shape", output.shape)
        return output


class SineCosineTimeEncoder(nn.Module):
    """Time encoder for encoding the elapsed time into a ``time_dim'' dimensional vector containing
    both sine and cosine transformations."""

    def __init__(
        self,
        time_dim: int,
        parameter_requires_grad: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        """
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()
        assert (
            time_dim % 2 == 0
        ), "time_dim must be an even number to split into equal number of sine and cosine transformations"
        self.time_dim = time_dim
        self.half_time_dim = time_dim // 2
        # trainable parameters for time encoding
        self.w = nn.Linear(1, self.half_time_dim)
        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.half_time_dim, dtype=np.float32))
            ).reshape(self.half_time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(self.half_time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False
        self.mean = mean
        self.std = std

    def forward(self, timestamps: torch.Tensor):
        """Compute time encodings of time in timestamps :param timestamps: Tensor, shape
        (batch_size, seq_len) :return:"""
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
        timestamps = (timestamps - self.mean) / self.std
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cat([torch.cos(self.w(timestamps)), torch.sin(self.w(timestamps))], dim=-1)
        return output


class TanhTimeEncoder(nn.Module):
    """Tanh Time encoder for encoding the elapsed time into a ``time_dim'' dimensional vector."""

    def __init__(self, time_dim: int):
        """
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.tanh = nn.Tanh()

    def forward(self, timestamps: torch.Tensor):
        """Compute time encodings of time in timestamps :param timestamps: Tensor, shape
        (batch_size, seq_len) :return:"""
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = self.tanh(self.w(timestamps))
        # print("output shape", output.shape)
        return output


class ExpTimeEncoder(nn.Module):
    def __init__(
        self, time_dim: int, median_inter_event_time: float, parameter_requires_grad: bool = True
    ):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(
                -np.log(np.linspace(0, 1, time_dim + 1, dtype=np.float32)[1:])
                / median_inter_event_time
            ).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """Compute time encodings of time in timestamps :param timestamps: Tensor, shape
        (batch_size, seq_len) :return:"""
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
        # Tensor, shape (batch_size, seq_len, time_dim)
        exponent = torch.clamp(-self.w(timestamps), max=40.0)  # avoid overflow
        output = torch.exp(exponent)
        return output


class NoTimeEncoder(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, timestamps: torch.Tensor):
        timestamps = timestamps.unsqueeze(dim=2)  # (batch_size, seq_len, 1)
        timestamps = (timestamps - self.mean) / self.std
        return timestamps


class LinearTimeEncoder(nn.Module):
    def __init__(
        self,
        time_dim: int,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__()
        self.no_time_encoder = NoTimeEncoder(mean=mean, std=std)
        self.linear = nn.Linear(1, time_dim)

    def forward(self, timestamps: torch.Tensor):
        timestamps = self.no_time_encoder(timestamps)
        return self.linear(timestamps)
