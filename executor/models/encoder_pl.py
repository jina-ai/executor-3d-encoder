from ctypes import DEFAULT_MODE
from typing import Optional

import pytorch_lightning as pl
import torch
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .modeling import MeshDataModel

AVAILABLE_MODELS = {
    'PointNet-Shapenet-d1024': {
        'model_name': 'pointnet',
        'hidden_dim': 1024,
        'embed_dim': 1024,
        'model_path': '',
    },
    'PointConv-Shapenet-d1024': {
        'model_name': 'pointconv',
        'hidden_dim': 1024,
        'embed_dim': 1024,
        'model_path': 'https://jina-pretrained-models.s3.us-west-1.amazonaws.com/mesh_models/pointconv-shapenet-d1024.pth',
    },
    'PointNet-Shapenet-d512': {
        'model_name': 'pointnet',
        'hidden_dim': 1024,
        'embed_dim': 512,
        'model_path': '',
    },
    'PointConv-Shapenet-d512': {
        'model_name': 'pointconv',
        'hidden_dim': 1024,
        'embed_dim': 512,
        'model_path': 'https://jina-pretrained-models.s3.us-west-1.amazonaws.com/mesh_models/pointconv-shapenet-d512.pth',
    },
}

DEFAULT_MODEL_NAME = 'pointconv'


class MeshDataEncoderPL(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = None,
        default_model_name=DEFAULT_MODEL_NAME,
        model_path: Optional[str] = None,
        hidden_dim: int = 1024,
        embed_dim: int = 1024,
        input_shape: str = 'bnc',
        device: str = 'cpu',
        batch_size: int = 64,
        filters: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        model_path = None
        if pretrained_model in AVAILABLE_MODELS:
            config = AVAILABLE_MODELS[pretrained_model]
            model_name = config.pop('model_name')
            model_path = config.pop('model_path')
            embed_dim = config.pop('embed_dim')
            hidden_dim = config.pop('hidden_dim')
        else:
            model_name = default_model_name
        self._model = MeshDataModel(
            model_name=model_name,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            pretrained=True if model_path else False,
            input_shape=input_shape,
        )

        if model_path:
            if model_path.startswith('http'):
                import os
                import urllib.request
                from pathlib import Path

                cache_dir = Path.home() / '.cache' / 'jina-models'
                cache_dir.mkdir(parents=True, exist_ok=True)

                file_url = model_path
                file_name = os.path.basename(model_path)
                model_path = cache_dir / file_name

                if not model_path.exists():
                    print(f'=> download {file_url} to {model_path}')
                    urllib.request.urlretrieve(file_url, model_path)

            checkpoint = torch.load(model_path, map_location='cpu')
            self._model.load_state_dict(checkpoint)

        self._device = device
        self._batch_size = batch_size
        self._filters = filters
        # bnc
        self.example_input_array = torch.zeros((batch_size, 1024, 3))

    def forward(self, x):
        embedding = self._model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60], gamma=0.5
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, _batch_idx):
        x, y = train_batch
        loss_fn = TripletLoss(
            miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
        )
        embeddings = self._model(x)
        loss = loss_fn(embeddings, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage):
        x, y = batch
        loss_fn = TripletLoss(
            miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
        )
        embeddings = self._model(x)
        loss = loss_fn(embeddings, y)
        self.log(f'{stage}_loss', loss, prog_bar=True)

    def validation_step(self, val_batch, _batch_idx):
        self.evaluate(val_batch, 'val')

    def test_step(self, test_batch, _batch_idx):
        self.evaluate(test_batch, 'test')
