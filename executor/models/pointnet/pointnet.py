import torch
import torch.nn as nn

from ..pooling import Pooling


class PointNet(nn.Module):
    def __init__(
        self,
        emb_dims=1024,
        input_shape='bnc',
        use_bn=True,
        global_feat=True,
        num_classes=40,
        classifier=False,
    ):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNet, self).__init__()
        if input_shape not in ['bcn', 'bnc']:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.use_bn = use_bn
        self.global_feat = global_feat
        if self.global_feat:
            self.pooling = Pooling('max')

        self.layers = self.create_model()
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dims, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.use_classifier = classifier

    def create_model(self):
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
        self.relu = torch.nn.ReLU()

        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.bn3 = torch.nn.BatchNorm1d(64)
            self.bn4 = torch.nn.BatchNorm1d(128)
            self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)

        if self.use_bn:
            layers = [
                self.conv1,
                self.bn1,
                self.relu,
                self.conv2,
                self.bn2,
                self.relu,
                self.conv3,
                self.bn3,
                self.relu,
                self.conv4,
                self.bn4,
                self.relu,
                self.conv5,
                self.bn5,
                self.relu,
            ]
        else:
            layers = [
                self.conv1,
                self.relu,
                self.conv2,
                self.relu,
                self.conv3,
                self.relu,
                self.conv4,
                self.relu,
                self.conv5,
                self.relu,
            ]
        return layers

    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == 'bnc':
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]
        if input_data.shape[1] != 3:
            raise RuntimeError('shape of x must be of [Batch x 3 x NumInPoints]')

        output = input_data
        for idx, layer in enumerate(self.layers):
            output = layer(output)
            if idx == 1 and not self.global_feat:
                point_feature = output

        # output = torch.max(output, 2, keepdim=True)[0]
        # output = output.view(-1, self.emb_dims)

        if self.global_feat:
            output = self.pooling(output)
            embedding = output
        else:
            # output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
            # output = self.pooling(output)
            output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
            embedding = torch.cat([output, point_feature], 1)

        if self.use_classifier:
            return self.classifier(embedding)
        else:
            return embedding
