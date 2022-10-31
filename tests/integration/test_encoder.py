import numpy as np
import pytest
import torch
from jina import DocumentArray, Flow
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from executor import MeshDataEncoder, MeshDataEncoderPL


@pytest.mark.parametrize(
    'model_name',
    ['pointconv', 'pointnet', 'pointnet2', 'curvenet', 'repsurf'],
)
def test_integration(model_name: str):
    docs = DocumentArray.empty(5)
    docs.tensors = np.random.random((5, 1024, 3))
    with Flow(return_results=True).add(
        uses=MeshDataEncoder,
        uses_with={'pretrained_model': None, 'default_model_name': model_name},
    ) as flow:
        resp = flow.post(
            on='/encoder',
            inputs=docs,
            return_results=True,
        )

        for doc in resp:
            assert doc.embedding is not None
            assert doc.embedding.shape == (1024,)


@pytest.mark.parametrize(
    'model_name, hidden_dim, embed_dim',
    [
        ('pointnet', 512, 512),
        ('pointnet2', 512, 512),
        ('curvenet', 512, 512),
        ('repsurf', 512, 512),
    ],
)
def test_integration_pytorch_lightning(
    model_name: str, hidden_dim, embed_dim, train_and_val_data, test_data
):
    encoder = MeshDataEncoderPL(
        default_model_name=model_name, hidden_dim=hidden_dim, embed_dim=embed_dim
    )

    train_data, validate_data = random_split(train_and_val_data, [120, 80])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    validate_loader = DataLoader(
        validate_data, batch_size=32, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)

    trainer = Trainer(
        accelerator='cpu',
        max_epochs=2,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )

    encoder.train()
    trainer.fit(encoder, train_loader, validate_loader)

    encoder.eval()
    trainer.test(encoder, dataloaders=test_loader)

    data = torch.from_numpy(np.random.random((5, 1024, 3)).astype(np.float32))
    embedding = encoder.forward(data)

    assert embedding is not None
    assert embedding.shape == (
        5,
        embed_dim,
    )
