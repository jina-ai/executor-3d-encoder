# 3D Mesh Encoder

An executor that loads 3D mesh models and embeds documents.


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://3DMeshEncoder')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://3DMeshEncoder')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`