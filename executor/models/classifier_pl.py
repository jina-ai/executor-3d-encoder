from typing import Optional

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .modeling import MeshDataModel, get_model

DEFAULT_MODEL_NAME = 'pointconv'


class MeshDataClassifierPL(pl.LightningModule):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        hidden_dim: int = 1024,
        input_shape: str = 'bnc',
        device: str = 'cpu',
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self._model = get_model(
            model_name=model_name,
            hidden_dim=hidden_dim,
            input_shape=input_shape,
            classifier=True,
        )

        self._device = device
        self._batch_size = batch_size
        # bnc
        self.example_input_array = torch.zeros((batch_size, 1024, 3))
        self._model_name = model_name

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        # optimizer and scheduler adapted from upstream
        if self._model_name == 'pointmlp':
            # 300 epochs
            optimizer = torch.optim.SGD(
                self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 300, eta_min=0.005, last_epoch=-1
            )
        elif self._model_name == 'repsurf':
            # 500 epochs
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.7
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60], gamma=0.5
            )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, _batch_idx):
        x, y = train_batch
        logits = self._model(x)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage):
        x, y = batch
        logits = self._model(x)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, val_batch, _batch_idx):
        self.evaluate(val_batch, 'val')

    def test_step(self, test_batch, _batch_idx):
        self.evaluate(test_batch, 'test')
