import numpy as np
import pytest
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, n_samples, n_points=1024, n_classes=40) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.points = np.random.random((n_samples, n_points, 3)).astype(np.float32)
        self.labels = np.sort(np.random.randint(n_classes, size=(n_samples,)))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return (
            self.points[index],
            self.labels[index],
        )


def create_torch_dataset(n_samples, n_points=1024, n_classes=40):
    return RandomDataset(n_samples, n_points, n_classes)


@pytest.fixture()
def train_and_val_data():
    return create_torch_dataset(200)


@pytest.fixture()
def test_data():
    return create_torch_dataset(200)
