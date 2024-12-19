import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
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
        # print("timestamps shape", timestamps.shape)
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))
        # print("output shape", output.shape)
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
        # print("timestamps shape", timestamps.shape)
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
        # if exponent.min() < -88:
        #     print("exponents", exponent, exponent.min())
        #     print("timestamps", timestamps)
        #     print("output", output)
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


# exp_time_encoder = TimeEncoder(time_dim=8)
# timestamps = torch.tensor([[3000,1000,10]], dtype=torch.float32)
# print(timestamps)
# output = exp_time_encoder(timestamps)
# print(output)

# exp_time_encoder = ExpTimeEncoder(time_dim=8, median_inter_event_time=1000)
# timestamps = torch.tensor([[3000,1000,10]], dtype=torch.float32)
# print(timestamps)
# output = exp_time_encoder(timestamps)
# print(output)

# no_time_encoder = NoTimeEncoder(mean=100, std=100)
# timestamps = torch.tensor([[3000,1000,10], [20, 30,100]], dtype=torch.float32)
# print(timestamps)
# output = no_time_encoder(timestamps)
# print(output)
