from torch import LongTensor, Tensor, nn

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, spoofed_weight: float = 1., bonafide_weight: float = 1.):
        super().__init__()
        self.loss_module = nn.CrossEntropyLoss(weight=Tensor([spoofed_weight, bonafide_weight]))

    def forward(self, pred_logits: Tensor, is_bonafide: LongTensor, **batch) -> Tensor:
        """
        :param pred_logits: (B, C)
        :param is_bonafide: (B,) with values 0/1
        """
        return self.loss_module(pred_logits, is_bonafide)
