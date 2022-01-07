__copyright__ = 'Copyright (c) 2022 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Optional

import numpy as np
import torch
from jina import Document, DocumentArray, Executor, requests

from .models import MeshDataModel

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
        'model_path': '',
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


def normalize(doc: 'Document'):
    points = doc.blob
    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale
    doc.blob = points.astype(np.float32)
    return doc


class MeshDataEncoder(Executor):
    """
    An executor that encodes 3D mesh data document.
    """

    def __init__(
        self,
        pretrained_model: str = 'PointConv-Shapenet-d512',
        default_model_name: str = 'pointconv',
        model_path: Optional[str] = None,
        hidden_dim: int = 1024,
        embed_dim: int = 1024,
        input_shape: str = 'bnc',
        device: str = 'cpu',
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        """
        :param pretrained_model: The pretrained model path.
        :param default_model_name: The name of the default model. Models listed on:
            https://github.com/jina-ai/executor-3d-encoder
        :param model_path: The path of the trained models checkpoint.
        :param emb_dims: The dimension of embeddings.
        :param input_shape: The shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        :param device: The device to use.
        :param batch_size: The batch size to use.
        """
        super().__init__(**kwargs)

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
            pretrained=False if model_path else True,
        )
        self._model.eval()

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

    @requests
    def encode(self, docs: DocumentArray, **_):
        """Encode docs."""
        docs.apply(normalize)
        docs.embed(
            self._model,
            device=self._device,
            batch_size=self._batch_size,
            to_numpy=True,
        )
