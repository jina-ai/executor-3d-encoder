import numpy as np
import pytest
from jina import Document, DocumentArray

from executor import MeshDataEncoder


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
def test_encoder(model_name, hidden_dim, embed_dim):
    encoder = MeshDataEncoder(
        pretrained_model=None,
        default_model_name=model_name,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    )

    docs = DocumentArray(Document(tensor=np.random.random((1024, 3))))

    encoder.encode(docs)

    assert docs[0].embedding is not None
    assert docs[0].embedding.shape == (embed_dim,)


def test_filter():
    encoder = MeshDataEncoder(
        pretrained_model=None,
        default_model_name='pointconv',
        filters={'embedding': {'$exists': False}},
    )

    docs = DocumentArray(Document(tensor=np.random.random((1024, 3))))

    embedding = np.random.random((512,))
    docs.append(Document(tensor=np.random.random((1024, 3)), embedding=embedding))

    encoder.encode(docs)

    assert docs[0].embedding.shape == (1024,)

    assert docs[1].embedding is not None
    assert docs[1].embedding.shape == (512,)
