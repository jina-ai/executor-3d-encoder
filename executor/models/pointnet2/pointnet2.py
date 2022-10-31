import torch.nn as nn
import torch.nn.functional as F

from ..pointconv import PointNetSetAbstraction, PointNetSetAbstractionMsg


class PointNet2(nn.Module):
    def __init__(
        self,
        emb_dims=1024,
        input_shape='bnc',
        normal_channel=True,
        classifier=False,
        num_classes=40,
        density_adaptive_type='ssg',
        pretrained=None,
    ):
        super(PointNet2, self).__init__()

        if input_shape not in ['bnc', 'bcn']:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )

        self.emb_dims = emb_dims
        self.input_shape = input_shape
        self.classifier = classifier
        self.normal_channel = normal_channel

        if density_adaptive_type == 'ssg':
            if normal_channel:
                in_channel = 6
            else:
                in_channel = 3
            self.sa1 = PointNetSetAbstraction(
                npoint=512,
                radius=0.2,
                nsample=32,
                in_channel=in_channel,
                mlp=[64, 64, 128],
                group_all=False,
            )
            self.sa2 = PointNetSetAbstraction(
                npoint=128,
                radius=0.4,
                nsample=64,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                group_all=False,
            )
            self.sa3 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256 + 3,
                mlp=[256, 512, self.emb_dims],
                group_all=True,
            )
        else:
            if normal_channel:
                in_channel = 3
            else:
                in_channel = 0
            self.sa1 = PointNetSetAbstractionMsg(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 128],
                in_channel=in_channel,
                mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2 = PointNetSetAbstractionMsg(
                npoint=128,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[32, 64, 128],
                in_channel=320,
                mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            self.sa3 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=640 + 3,
                mlp=[256, 512, self.emb_dims],
                group_all=True,
            )

        self.fc1 = nn.Linear(self.emb_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        if self.input_shape == 'bnc':
            xyz = xyz.permute(0, 2, 1)
            batch_size = xyz.shape[0]
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(batch_size, self.emb_dims)

        if self.classifier:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            output = F.log_softmax(x, -1)
        else:
            output = x

        return output
