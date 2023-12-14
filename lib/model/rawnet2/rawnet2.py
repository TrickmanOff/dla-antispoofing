from typing import Any, Dict, Optional, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from lib.model.base_model import BaseModel
from .res_block import ResBlocksStack
from .sinc_conv import SincConv_fast


class RawNet2(BaseModel):
    def __init__(self, sinc_conv_config: Optional[Dict[str, Any]] = None,
                 res_blocks_out_channels: Sequence[int] = (128, 128, 512, 512, 512, 512),
                 normalize_before_gru: bool = False,
                 gru_hidden_size: int = 1024,
                 gru_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        sinc_conv_config = sinc_conv_config or {}
        gru_config = gru_config or {}

        # input - of shape (B, 1, T)
        self.sinc_block = nn.Sequential(
            SincConv_fast(out_channels=128, **sinc_conv_config),
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        # output - of shape (B, 128, T')

        self.res_blocks = ResBlocksStack(in_channels=128, out_channels=res_blocks_out_channels)

        # input of shape (B, T, C)
        if normalize_before_gru:
            self.norm_before_gru = nn.Sequential(
                nn.BatchNorm1d(res_blocks_out_channels[-1]),
                nn.LeakyReLU(),
            )
        else:
            self.norm_before_gru = None
        self.gru = nn.GRU(input_size=res_blocks_out_channels[-1], batch_first=True, hidden_size=gru_hidden_size, **gru_config)

        self.head = nn.Linear(gru_hidden_size, 2)

    def forward(self, wave: Tensor) -> Tensor:
        """
        :param wave: (B, 1, T)
        :return: probs: (B, 2)
        """
        output = self.sinc_block(wave)  # (B, C, T')
        output = self.res_blocks(output)  # (B, C', T'')
        if self.norm_before_gru is not None:
            output = self.norm_before_gru(output)
        output = self.gru(output.transpose(-2, -1))[0][:, -1]  # (B, hidden_size)
        output = self.head(output)  # (B, 2)
        probs = F.softmax(output, dim=-1)
        return probs
