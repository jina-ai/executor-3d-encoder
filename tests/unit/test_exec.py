import numpy as np
import pytest
from jina import Document, DocumentArray

from executor import MeshDataEncoder


@pytest.mark.parametrize('model_name', ['pointconv', 'pointnet'])
def test_encoder(model_name):
    encoder = MeshDataEncoder(pretrained_model=None, default_model_name=model_name)

    docs = DocumentArray(Document(tensor=np.random.random((1024, 3))))

    encoder.encode(docs)

    assert docs[0].embedding is not None
    assert docs[0].embedding.shape == (1024,)
