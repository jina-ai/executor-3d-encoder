import numpy as np
import pytest
from jina import DocumentArray, Flow
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from executor import MeshDataEncoder, MeshDataEncoderPL
from tests.conftest import create_torch_dataset


@pytest.mark.parametrize(
    'model_name, hidden_dim, embed_dim',
    [
        ('pointconv', 1024, 1024),
        ('pointnet', 1024, 1024),
        ('pointnet2', 1024, 1024),
        ('pointmlp', 64, 32),
        ('repsurf', 1024, 1024),
    ],
)
def test_integration(model_name: str, hidden_dim: int, embed_dim: int):
    docs = DocumentArray.empty(5)
    docs.tensors = np.random.random((5, 1024, 3))
    with Flow(return_results=True).add(
        uses=MeshDataEncoder,
        uses_with={
            'pretrained_model': None,
            'default_model_name': model_name,
            'hidden_dim': hidden_dim,
            'embed_dim': embed_dim,
        },
    ) as flow:
        resp = flow.post(
            on='/encoder',
            inputs=docs,
            return_results=True,
        )

        for doc in resp:
            assert doc.embedding is not None
            assert doc.embedding.shape == (embed_dim,)


@pytest.fixture(name='create_torch_dataset')
@pytest.mark.parametrize(
    'model_name, hidden_dim, embed_dim',
    [
        ('pointconv', 1024, 1024),
        ('pointnet', 1024, 1024),
        ('pointnet2', 1024, 1024),
        ('pointmlp', 64, 32),
        ('repsurf', 1024, 1024),
    ],
)
def test_integration_pytorch_lightning(
    model_name: str, hidden_dim: int, embed_dim: int, create_torch_dataset
):
    encoder = MeshDataEncoderPL(
        default_model_name=model_name, hidden_dim=hidden_dim, embed_dim=embed_dim
    )

    train_and_val_data = create_torch_dataset(10, 1024)
    test_data = create_torch_dataset(4, 1024)

    train_data, validate_data = random_split(train_and_val_data, [4, 1])

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

    trainer = Trainer(
        accelerator='cpu',
        max_epochs=5,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )

    encoder.train()
    trainer.fit(encoder, train_loader, validate_loader)

    encoder.eval()
    trainer.test(encoder, dataloaders=test_loader)

    data = np.random.random((5, 1024, 3))
    embedding = encoder.forward(data)

    assert embedding is not None
    assert embedding.shape == (
        5,
        embed_dim,
    )

