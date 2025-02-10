import torch
import pytorch_lightning as pl

import torch.nn as nn

# TODO adaptar para os casos 1d de forma a ser n-d
class EMDLoss(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        if len(p.shape)==4: # TODO implementar de forma mais genérica com única exeção o cenário 1d
            p1, p2 = torch.sum(p, dim=-1), torch.sum(p, dim=-2)
            q1, q2 = torch.sum(q, dim=-1), torch.sum(q, dim=-2)

            x1 = p1 - q1
            y1 = torch.cumsum(x1, dim=0)

            x2 = p2 - q2
            y2 = torch.cumsum(x2, dim=0)

            return torch.abs(y1).sum() + torch.abs(y2).sum()
        else:
            x = p - q
            y = torch.cumsum(x, dim=0)
            return torch.abs(y).sum()