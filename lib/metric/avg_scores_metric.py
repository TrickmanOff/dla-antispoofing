from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import LongTensor, Tensor

from lib.metric.base_metric import BaseMetric


class AverageScoresMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_metrics_names(self) -> List[str]:
        return [f'A{i:02d}' for i in range(1, 20)] + ['bonafide']

    def __call__(self, pred_logits: Tensor, is_bonafide: LongTensor, spoofing_algo: List[str], **batch) -> Union[float, Dict]:
        scores = F.softmax(pred_logits.detach().cpu(), dim=-1)[:, 1]  # (B,)
        labels = np.array(spoofing_algo)
        labels = np.where(labels == '-', 'bonafide', labels)

        df = pd.DataFrame({'type': labels, 'score': scores})
        mean_scores = df.groupby('type').mean().to_dict()['score']

        return mean_scores
