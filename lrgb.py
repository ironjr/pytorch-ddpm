import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import AutoEncoder


class LRGBLoss(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        loss_type: str = 'L1',
        remap: dict = None,
    ) -> None:
        super(LRGBLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.remap = remap

        self.encoder = AutoEncoder(20)
        self.encoder.load_state_dict(torch.load(checkpoint))
        print(checkpoint)
        for p in self.parameters():
            p.requires_grad = False

        if self.loss_type in ('l1', 'mae'):
            self.dist = F.l1_loss
        elif self.loss_type in ('l2', 'mse'):
            self.dist = F.mse_loss
        elif self.loss_type in ('huber', 'smoothl1', 'sl1'):
            self.dist = F.smooth_l1_loss
        else:
            raise ValueError(
                'Unknown loss type [{:s}] is detected'.format(loss_type))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Loss={self.loss_type})"

    @torch.cuda.amp.autocast(False)
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.remap is not None:
            x = kwargs[self.remap['x']]
            y = kwargs[self.remap['y']]
        x_feat = self._get_features(x)
        with torch.no_grad():
            y_feat = self._get_features(y)
        return self.dist(x_feat, y_feat)

    def state_dict(self, *args, **kwargs):
        return {}

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)


def instantiate(opt: dict, loss_opt: dict):
    kwargs = {
        #  'checkpoint': '../pretrained/model_latest.pt',
        #  'checkpoint': '../../LRGB/experiment/div2k_long/model/model_latest.pt',
        'checkpoint': '../lrgb_pretrained/FINAL/idf8_ndb-lbl2-lr5e-4-coslr1k-c20/model/model_latest.pt',
        'loss_type': 'l1',
        'remap': None,
    }
    for k in kwargs.keys():
        if k in loss_opt:
            kwargs[k] = loss_opt[k]
    loss = LRGBLoss(**kwargs)
    return loss
