# 3D Mesh Encoder

An Executor that receives Documents containing point sets data in its `tensor` attribute, with shape `(N, 3)` and encodes it to embeddings of shape `(D,)`.
Now, the following pretrained models are ready to be used to create embeddings:

- **PointConv-Shapenet-d512**: A **PointConv** model resulted in **512** dimension of embeddings, which is finetuned based on ShapeNet dataset.
- **PointConv-Shapenet-d1024**: A **PointConv** model resulted in **1024** dimension of embeddings, which is finetuned based on ShapeNet dataset.



## Usage

#### via Docker image (recommended)

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://MeshDataEncoder', \
               uses_with={'pretrained_model': 'PointConv-Shapenet-d512'})
```

#### via source code

```python
from jina import Flow

f = Flow().add(uses='jinahub://MeshDataEncoder', \
               uses_with={'pretrained_model': 'PointConv-Shapenet-d512'})
```

This Executor offers a GPU tag to speed up encoding. For more information on how to run the executor on GPU, check out the documentation.


## How to finetune pretrained-model?

### Finetune pretrained-model with finetuner
#### install finetuner

```bash
$ pip install finetuner
```

#### prepare dataset

TBD...

#### finetuning model with labeled dataset

```bash
$ python finetune.py --help

$ python finetune.py --model_name pointconv \
    --train_dataset /path/to/train.bin \
    --eval_dataset /path/to/eval.bin \
    --batch_size 128 \
    --epochs 50
```

#### finetuning model with unlabeled dataset

```bash
$ python finetune.py --model_name pointconv \
    --train_dataset /path/to/unlabeled_data.bin \
    --interactive
```

### Finetune pretrained-model with Pytorch Lightning
#### prepare dataset

To use your customized dataset, you should design your own dataset code, like those in `datasets/` directory. Here `datasets/modelnet40.py` is an example:

```python
class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, data_path, sample_points=1024, seed=10) -> None:
        super().__init__()
        # extract point data and labels from your file, e.g. npz, h5, etc.
        data = np.load(data_path)
        self.points = data['tensor']
        self.labels = data['labels']
        self.sample_points = sample_points

    def __len__(self):
        # return the total length of your data
        return len(self.labels)

    def __getitem__(self, index):
        return (
            # process on the fly, if needed
            preprocess(self.points[index], num_points=self.sample_points),
            self.labels[index],
        )
```

#### finetuning model with labeled dataset

```bash
$ python finetune_pl.py --help

$ python finetune_pl.py --model_name pointconv \
    --train_dataset /path/to/train.bin \
    --eval_dataset /path/to/eval.bin \
    --split_ratio 0.8 \
    --checkpoint_path /path/to/checkpoint/ \
    --embed_dim 512 \
    --hidden_dim 1024 \
    --batch_size 128 \
    --epochs 50
```

## References

- [PointNet](https://arxiv.org/abs/1612.00593):  Deep Learning on Point Sets for 3D Classification and Segmentation
- [PointConv](https://arxiv.org/abs/1811.07246): Deep Convolutional Networks on 3D Point Clouds
- [PointNet++](http://arxiv.org/abs/1706.02413): PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- [PointMLP](http://arxiv.org/abs/2202.07123): Rethinking Network Design and Local Geometry in Point Cloud
- [RepSurf](https://arxiv.org/abs/2205.05740): Surface Representation for Point Clouds
- [CurveNet](https://arxiv.org/abs/2105.01288): Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis
