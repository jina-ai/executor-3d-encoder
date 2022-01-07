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

## How to finetune model?

### install finetuner

```bash
$ pip install finetuner
```

### prepare dataset

TBD...

### finetuning model with labeled dataset

```bash
$ python finetune.py --help

$ python finetune.py --model_name pointconv --train_dataset /path/to/train.bin --eval_dataset /path/to/eval.bin
```

### finetuning model with unlabeled dataset

```bash
$ python finetune.py --model_name pointconv --train_dataset /path/to/unlabeled_data.bin --interactive
```
