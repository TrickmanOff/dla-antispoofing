from typing import Dict, Union

from torch import LongTensor, Tensor

from lib.metric.base_metric import BaseMetric
from lib.metric.utils import compute_eer


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER)
    """
    def __init__(self, return_rates: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_rates = return_rates

    def __call__(self, pred_logits: Tensor, is_bonafide: LongTensor, **batch) -> Union[float, Dict]:
        if self.return_rates:
            eer, thres, frr, far = compute_eer(pred_logits, is_bonafide, return_rates=True)
            return {
                'metric': eer,
                'eer': eer,
                'frr': frr,
                'far': far,
            }
        else:
            eer, thres = compute_eer(pred_logits, is_bonafide, return_rates=False)
            return eer
