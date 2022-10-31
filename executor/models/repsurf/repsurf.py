"""ref: https://github.com/hancyran/RepSurf/blob/main/models/repsurf/scanobjectnn/repsurf_ssg_umb.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .repsurf_utils import SurfaceAbstractionCD, UmbrellaSurfaceConstructor


class RepSurf(torch.nn.Module):
    def __init__(
        self,
        num_points,
        return_center=True,
        return_polar=True,
        return_dist=True,
        group_size=8,
        umb_pool_type='sum',
        num_classes=40,
        input_shape='bnc',
        emb_dims=1024,
        classifier=False,
    ) -> None:
        super(RepSurf, self).__init__()

        if input_shape not in ['bnc', 'bcn']:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape

        center_channel = 0 if not return_center else (6 if return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = num_points
        self.return_dist = return_dist
        self.emb_dims = emb_dims

        self.surface_constructor = UmbrellaSurfaceConstructor(
            group_size + 1,
            repsurf_channel,
            return_dist=return_dist,
            aggr_type=umb_pool_type,
            cuda=False,
        )

        self.sa1 = SurfaceAbstractionCD(
            npoint=512,
            radius=0.2,
            nsample=32,
            feat_channel=repsurf_channel,
            pos_channel=center_channel,
            mlp=[64, 64, 128],
            group_all=False,
            return_polar=return_polar,
            cuda=False,
        )

        self.sa2 = SurfaceAbstractionCD(
            npoint=128,
            radius=0.4,
            nsample=64,
            feat_channel=128 + repsurf_channel,
            pos_channel=center_channel,
            mlp=[128, 128, 256],
            group_all=False,
            return_polar=return_polar,
            cuda=False,
        )

        self.sa3 = SurfaceAbstractionCD(
            npoint=None,
            radius=None,
            nsample=None,
            feat_channel=256 + repsurf_channel,
            pos_channel=center_channel,
            mlp=[256, 512, self.emb_dims],
            group_all=True,
            return_polar=return_polar,
            cuda=False,
        )

        # modelnet40
        self.use_classifier = classifier
        self.classfier = nn.Sequential(
            nn.Linear(self.emb_dims, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        if self.input_shape == 'bnc':
            points = points.permute(0, 2, 1)

        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)

        feature = feature.view(-1, self.emb_dims)

        if self.use_classifier:
            feature = self.classfier(feature)
            feature = F.log_softmax(feature, -1)

        return feature
