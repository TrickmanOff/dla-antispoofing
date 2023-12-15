from typing import Sequence

from torch import LongTensor, Tensor, nn

from .fms import FeatureMapScaling


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.maxpooling_size = 3

        self.pre_res = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
        )
        self.scaling_input = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.post_res = nn.Sequential(
            nn.MaxPool1d(self.maxpooling_size),
            FeatureMapScaling(out_channels),
        )

    def transform_input_lengths(self, input_lengths: LongTensor) -> LongTensor:
        return input_lengths // self.maxpooling_size

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (B, C=in_channels, num_features)
        :return: (B, C'=out_channels, num_features')
        """
        return self.post_res(self.pre_res(input) + self.scaling_input(input))


class ResBlocksStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: Sequence[int] = (128, 512)):
        super().__init__()
        blocks = nn.ModuleList()
        norms = nn.ModuleList()
        for i, num_out_channels in enumerate(out_channels):
            if i != 0:
                norm = nn.Sequential(
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(),
                )
                norms.append(norm)
            blocks.append(ResBlock(in_channels, num_out_channels))
            in_channels = num_out_channels
        self.blocks = blocks
        self.norms = norms

    def transform_input_lengths(self, input_lengths: LongTensor) -> LongTensor:
        for block in self.blocks:
            input_lengths = block.transform_input_lengths(input_lengths)
        return input_lengths

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (B, C=in_channels, num_features)
        :return: (B, C', num_features')
        """
        for i, block in enumerate(self.blocks):
            input = block(input)
            if i < len(self.norms):
                input = self.norms[i](input)
        return input
