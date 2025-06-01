import torch.nn as nn
import torch
import torch.nn.functional as F

class LabelSmoothing(nn.Module):

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction='none')
        self.register_buffer('true_dist', None)
        if smoothing < 0 or smoothing > 1:
            raise ValueError("smoothing: от 0 до 1")
        if padding_idx < 0:
            raise ValueError("padding_idx должен быть полож.")

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.size:
            raise ValueError(f"нужен размер словаря {self.size}, переданный размер {x.size(1)}")
        true_dist = torch.full_like(x, self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        padding_mask = target == self.padding_idx
        if padding_mask.any():
            true_dist[padding_mask] = 0.0
        self.true_dist = true_dist
        log_probs = F.log_softmax(x, dim=1)
        losses = self.criterion(log_probs, true_dist.detach())
        losses = losses.sum(dim=1)
        losses = losses.masked_fill(padding_mask, 0.0)
        if self.reduction == 'mean':
            non_padding_elements = (~padding_mask).sum()
            return losses.sum() / max(1, non_padding_elements)
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses