import torch
from torch import nn

from .curvenet import CurveNet
from .pointconv import MLP, PointConv
from .pointmlp import pointMLP, pointMLPElite
from .pointnet import PointNet
from .pointnet2 import PointNet2
from .repsurf import RepSurf

PRETRAINED_MODELS = {
    'pointnet': {
        'model_path': '',
        'hidden_dim': 1024,
    },
    'pointconv': {
        'model_path': 'https://jina-pretrained-models.s3.us-west-1.amazonaws.com/mesh_models/pointconv_class_encoder.pth',
        'hidden_dim': 1024,
    },
}


def get_model(model_name: str, hidden_dim: int, input_shape: str, classifier: bool):
    if model_name == 'pointnet':
        # classifier ignored
        return PointNet(
            emb_dims=hidden_dim,
            input_shape=input_shape,
            use_bn=True,
            global_feat=True,
            classifier=classifier,
        )
    elif model_name == 'pointconv':
        return PointConv(
            emb_dims=hidden_dim,
            input_channel_dim=3,
            input_shape=input_shape,
            classifier=classifier,
        )
    elif model_name == 'pointnet2':
        return PointNet2(
            emb_dims=hidden_dim,
            normal_channel=False,
            input_shape=input_shape,
            classifier=classifier,
            density_adaptive_type='ssg',
        )
    elif model_name == 'pointnet2msg':
        return PointNet2(
            emb_dims=hidden_dim,
            normal_channel=False,
            input_shape=input_shape,
            classifier=classifier,
            density_adaptive_type='msg',
        )
    elif model_name == 'repsurf':
        return RepSurf(
            num_points=1024,
            emb_dims=hidden_dim,
            input_shape=input_shape,
            classifier=classifier,
        )
    elif model_name == 'pointmlp':
        return pointMLP(classifier=classifier, embed_dim=hidden_dim)
    elif model_name == 'pointmlp-elite':
        return pointMLPElite(classifier=classifier, embed_dim=hidden_dim)
    elif model_name == 'curvenet':
        return CurveNet(
            emb_dims=hidden_dim,
            input_shape=input_shape,
            classifier=classifier,
        )
    else:
        raise NotImplementedError('The model has not been implemented yet!')


class MeshDataModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'pointnet',
        hidden_dim: int = 1024,
        embed_dim: int = 512,
        input_shape: str = 'bnc',
        dropout_rate: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        model_path = None
        if pretrained and model_name in PRETRAINED_MODELS:
            config = PRETRAINED_MODELS[model_name]
            model_path = config['model_path']
            hidden_dim = config['hidden_dim']

        self._point_encoder = get_model(model_name, hidden_dim, input_shape, False)

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

            print(f'==> restore {model_name} from: {model_path}')
            checkpoint = torch.load(model_path, map_location='cpu')
            self._point_encoder.load_state_dict(checkpoint)

        self._dropout = nn.Dropout(dropout_rate)

        # Projector
        self._projector = MLP(hidden_dim, hidden_dim * 4, embed_dim)

    @property
    def encoder(self):
        return self._point_encoder

    def forward(self, points):
        feats = self._point_encoder(points)
        feats = self._dropout(feats)
        return self._projector(feats)
