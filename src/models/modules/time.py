import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Time encoder for encoding the elapsed time into a ``time_dim'' dimensional vector."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
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

    def forward(self, timestamps: torch.Tensor):
        """Compute time encodings of time in timestamps :param timestamps: Tensor, shape
        (batch_size, seq_len) :return:"""
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
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
