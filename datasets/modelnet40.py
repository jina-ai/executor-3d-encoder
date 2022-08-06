import numpy as np
from torch.utils.data import Dataset

from preprocess import preprocess


class ModelNet40(Dataset):
    def __init__(self, data_path, sample_points=1024, seed=10) -> None:
        super().__init__()
        data = np.load(data_path)
        self.points = data['tensor']
        self.labels = data['labels']
        self.sample_points = sample_points

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            preprocess(self.points[index], num_points=self.sample_points),
            self.labels[index],
        )
