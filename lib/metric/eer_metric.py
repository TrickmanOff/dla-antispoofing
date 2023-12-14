import torch.nn.functional as F
from torch import LongTensor, Tensor

from lib.metric.base_metric import BaseMetric
from lib.metric.utils import compute_eer


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER)
    """
    def __call__(self, pred_logits: Tensor, is_bonafide: LongTensor, **batch) -> float:
        eer, thres = compute_eer(pred_logits, is_bonafide)
        return eer

