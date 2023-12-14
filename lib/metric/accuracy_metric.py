import torch.nn.functional as F
from torch import LongTensor, Tensor

from lib.metric.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Mean accuracy
    """
    def __init__(self, bonafide_threshold: float = 0.5):
        """
        :param bonafide_threshold: objects with scores >= 'bonafide_threshold' are considered positive
        """
        super().__init__()
        self.bonafide_threshold = bonafide_threshold

    def __call__(self, pred_logits: Tensor, is_bonafide: LongTensor, **batch) -> float:
        pred_logits = pred_logits.detach().cpu()
        is_bonafide = is_bonafide.detach().cpu()
        pred_probs = F.softmax(pred_logits, dim=-1)  # (B, 2)
        pred_is_bonafide = pred_probs[:, 1] >= self.bonafide_threshold  # (B,)
        return (pred_is_bonafide.long() == is_bonafide).float().mean().item()
