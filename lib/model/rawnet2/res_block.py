from typing import Sequence

from torch import Tensor, nn

from .fms import FeatureMapScaling


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.pre_res = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
        )
        self.scaling_input = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.post_res = nn.Sequential(
            nn.MaxPool1d(3),
            FeatureMapScaling(out_channels),
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (B, C=in_channels, num_features)
        :return: (B, C'=out_channels, num_features')
        """
        return self.post_res(self.pre_res(input) + self.scaling_input(input))


class ResBlocksStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: Sequence[int] = (128, 512)):
        super().__init__()
        blocks = []
        for i, num_out_channels in enumerate(out_channels):
            if i != 0:
                blocks += [
                    nn.BatchNorm1d(in_channels),
                    nn.LeakyReLU(),
                ]
            blocks.append(ResBlock(in_channels, num_out_channels))
            in_channels = num_out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (B, C=in_channels, num_features)
        :return: (B, C', num_features')
        """
        return self.blocks(input)
