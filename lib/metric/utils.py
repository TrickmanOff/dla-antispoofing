import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import LongTensor, Tensor

from lib.metric.calculate_eer import compute_eer as np_compute_eer


def compute_eer(pred_logits: Tensor, is_bonafide: LongTensor, return_rates: bool = False):
    """
    returns eer, threshold
    + if `return_rates` == True: frr, far
    """
    pred_logits = pred_logits.detach().cpu()
    pred_scores = F.softmax(pred_logits, dim=-1).numpy()  # (B, 2)
    is_bonafide = is_bonafide.detach().cpu().numpy()

    is_bonafide_mask = (is_bonafide == 1)
    bonafide_scores = pred_scores[is_bonafide_mask, 0]
    other_scores = pred_scores[~is_bonafide_mask, 0]

    return np_compute_eer(bonafide_scores, other_scores, return_rates=return_rates)


def basic_plot_eer(eer, frr, far) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    xs = np.arange(len(far))
    ax.plot(xs, far, label='FAR')
    ax.plot(xs, frr, label='FRR')
    ax.axhline(eer, label=f'EER: {eer:.4f}', color='red', linestyle='--')
    ax.get_xaxis().set_visible(False)

    ax.legend()
    return fig


def plot_eer(pred_logits: Tensor, is_bonafide: LongTensor) -> plt.Figure:
    eer, _, frr, far = compute_eer(pred_logits, is_bonafide, return_rates=True)
    return basic_plot_eer(eer, frr, far)
