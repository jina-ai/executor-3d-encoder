# 3D Mesh Encoder

An Executor that receives Documents containing point sets data in its `tensor` attribute, with shape `(N, 3)` and encodes it to embeddings of shape `(D,)`.
Now, the following pretrained models are ready to be used to create embeddings:

- **PointConv-Shapenet-d512**: A **PointConv** model resulted in **512** dimension of embeddings, which is finetuned based on ShapeNet dataset.
- **PointConv-Shapenet-d1024**: A **PointConv** model resulted in **1024** dimension of embeddings, which is finetuned based on ShapeNet dataset.



## Usage

#### via Docker image (recommended)

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://3DMeshEncoder', \
               uses_with={'pretrained_model': 'PointConv-Shapenet-d512'})
```

#### via source code

```python
from jina import Flow

f = Flow().add(uses='jinahub://3DMeshEncoder', \
               uses_with={'pretrained_model': 'PointConv-Shapenet-d512'})
```

This Executor offers a GPU tag to speed up encoding. For more information on how to run the executor on GPU, check out the documentation.


## How to finetune pretrained-model?

### install finetuner

```bash
$ pip install finetuner
```

### prepare dataset

TBD...

### finetuning model with labeled dataset

```bash
$ python finetune.py --help

$ python finetune.py --model_name pointconv \
    --train_dataset /path/to/train.bin \
    --eval_dataset /path/to/eval.bin \
    --batch_size 128 \
    --epochs 50
```

### finetuning model with unlabeled dataset

```bash
$ python finetune.py --model_name pointconv \
    --train_dataset /path/to/unlabeled_data.bin \
    --interactive
```


## References

- [PointNet](https://arxiv.org/abs/1612.00593):  Deep Learning on Point Sets for 3D Classification and Segmentation
- [PointConv](https://arxiv.org/abs/1811.07246): Deep Convolutional Networks on 3D Point Clouds
