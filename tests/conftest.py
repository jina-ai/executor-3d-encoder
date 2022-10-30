import numpy as np
import pytest
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, n_samples, n_points=1024, n_classes=40) -> None:
        super().__init__()
        self.points = np.random.random((n_samples, n_points, 3))
        self.labels = np.random.randint(n_classes, size=(n_samples))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.points[index, :, :],
            self.labels[index],
        )


def create_torch_dataset(n_samples, n_points, n_classes=40):
    return RandomDataset(n_samples, n_points, n_classes)
