from typing import Optional

import numpy as np
import torch
from jina import Document, DocumentArray, Executor, requests

from models import PointConv, PointNet

AVAILABLE_MODELS = {
    'PointNet-Shapenet-d1024': {
        'model_name': 'pointnet',
        'emb_dims': 1024,
        'model_path': '',
    },
    'PointConv-Shapenet-d1024': {
        'model_name': 'pointconv',
        'emb_dims': 1024,
        'model_path': '',
    },
    'PointNet-Shapenet-d512': {
        'model_name': 'pointnet',
        'emb_dims': 512,
        'model_path': '',
    },
    'PointConv-Shapenet-d512': {
        'model_name': 'pointconv',
        'emb_dims': 512,
        'model_path': '',
    },
}


def normalize(doc: 'Document'):
    points = np.transpose(doc.blob)
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
        pretrained_model: str = 'PointConv-Shapenet-d1024',
        default_model_name: str = 'pointconv',
        model_path: Optional[str] = None,
        emb_dims: Optional[int] = None,
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

        config = {}
        if pretrained_model in AVAILABLE_MODELS:
            config = AVAILABLE_MODELS[pretrained_model]
            model_name = config.pop('model_name')
            model_path = config.pop('model_path')

        else:
            model_name = default_model_name
            config['emb_dims'] = emb_dims

        self._encoder = {'pointnet': PointNet, 'pointconv': PointConv}[model_name](
            **config
        )
        self._encoder.eval()

        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self._encoder.load_state_dict(checkpoint)

        self._device = device
        self._batch_size = batch_size

    @requests
    def encode(self, docs: DocumentArray, **_):
        """Encode docs."""
        docs.apply(normalize)
        docs.embed(
            self._encoder,
            device=self._device,
            batch_size=self._batch_size,
            to_numpy=True,
        )
