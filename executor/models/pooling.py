import torch


class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
        super(Pooling, self).__init__()

        self.pool_type = pool_type

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 2).contiguous()
