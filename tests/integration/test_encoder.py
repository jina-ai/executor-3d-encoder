import numpy as np
import pytest
from jina import DocumentArray, Flow

from executor import MeshDataEncoder


@pytest.mark.parametrize('model_name', ['pointconv', 'pointnet'])
def test_integration(model_name: str):
    docs = DocumentArray.empty(5)
    docs.tensors = np.random.random((5, 1024, 3))
    with Flow(return_results=True).add(
        uses=MeshDataEncoder,
        uses_with={'pretrained_model': None, 'default_model_name': model_name},
    ) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            return_results=True,
        )

        for doc in resp:
            assert doc.embedding is not None
            assert doc.embedding.shape == (1024,)
