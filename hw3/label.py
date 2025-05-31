import torch.nn as nn
import torch

class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        true_dist = pred.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        ignore = target == self.padding_idx
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(ignore, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(pred, true_dist) / torch.sum(target != self.padding_idx)