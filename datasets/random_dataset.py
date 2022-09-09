import numpy as np
from torch.utils.data import Dataset

from preprocess import preprocess


class RandomDataset(Dataset):
    def __init__(self, npoints, n_classes=40) -> None:
        super().__init__()
        self.points = np.random.random((npoints, 1024, 3))
        self.labels = np.random.randint(n_classes, size=(npoints, 1024, 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.points,
            self.labels[index],
        )
