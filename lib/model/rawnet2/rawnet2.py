from typing import Any, Dict, Optional, Sequence

import torch
from torch import LongTensor, Tensor, nn

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

        self.sinc_maxpooling_size = 3

        # input - of shape (B, 1, T)
        self.sinc_block = nn.Sequential(
            SincConv_fast(**sinc_conv_config),
            nn.MaxPool1d(kernel_size=self.sinc_maxpooling_size),
            nn.BatchNorm1d(sinc_conv_config['out_channels']),
            nn.LeakyReLU(),
        )
        # freeze SincConv
        for param in self.sinc_block[0].parameters():
            param.requires_grad = False
        # output - of shape (B, sinc_conv_config['out_channels'], T')

        self.res_blocks = ResBlocksStack(in_channels=sinc_conv_config['out_channels'], out_channels=res_blocks_out_channels)

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

    def transform_input_lengths(self, input_lengths: LongTensor) -> LongTensor:
        # before gru

        # maxpooling in sinc
        input_lengths = self.sinc_block[0].transform_input_lengths(input_lengths)
        input_lengths = input_lengths // self.sinc_maxpooling_size
        # resblocks
        input_lengths = self.res_blocks.transform_input_lengths(input_lengths)

        return input_lengths

    def forward(self, wave: Tensor, wave_length: LongTensor, **batch) -> Tensor:
        """
        :param wave: (B, 1, T)
        :return: logits: (B, 2)
        """
        output = self.sinc_block(wave)  # (B, C, T')
        output = self.res_blocks(output)  # (B, C', T'')
        if self.norm_before_gru is not None:
            output = self.norm_before_gru(output)
        outputs = self.gru(output.transpose(-2, -1))[0]  # (B, seq_len, hidden_size)
        wave_length = self.transform_input_lengths(wave_length)
        output = outputs[torch.arange(outputs.shape[0]), wave_length - 1]  # (B, hidden_size)
        logits = self.head(output)  # (B, 2)
        return logits
