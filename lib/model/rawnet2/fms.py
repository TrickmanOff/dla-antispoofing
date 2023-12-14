import torch.nn.functional as F
from torch import Tensor, nn


class FeatureMapScaling(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        # input - of shape (B, C, T)
        self.scales_fc = nn.Linear(num_channels, num_channels)

    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: of shape (B, C, T)
        :return: of shape (B, C, T)
        """
        scales = F.avg_pool1d(features, kernel_size=features.shape[-1]).squeeze(-1)  # (B, C)
        scales = F.sigmoid(self.scales_fc(scales)).unsqueeze(-1)  # (B, C, 1)
        output = features * scales + scales  # mul-add FMS
        return output
