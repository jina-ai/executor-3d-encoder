__copyright__ = 'Copyright (c) 2022 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from functools import partial

import click
import finetuner
import numpy as np
import torch
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner
from jina import Document, DocumentArray

from models import MeshDataModel


def random_sample(pc, num):
    permutation = np.arange(len(pc))
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc


def preprocess(doc: 'Document', num_points: int = 1024, data_aug: bool = True):
    points = random_sample(doc.blob, num_points)
    points = np.transpose(points)

    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale

    if data_aug:
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
        points += np.random.normal(0, 0.02, size=points.shape)  # random jitter
    return points


@click.command()
@click.option('--train_dataset', help='The training dataset file path')
@click.option('--eval_dataset', help='The evaluation dataset file path')
@click.option('--embed_dim', default=1024, help='The embedding dimension')
@click.option('--checkpoint', help='The pretrained checkpoint')
@click.option('--dump_path', help='The target dump path of checkpoint')
@click.option('--model_name', default='pointnet', help='The model name')
@click.option('--batch_size', default=64, help='The pretrained clip model path')
@click.option('--epochs', default=50, help='The pretrained clip model path')
@click.option('--num_gpu', default=0, help='The number of GPUs')
def main(
    train_dataset,
    eval_dataset,
    model_name,
    embed_dim,
    checkpoint,
    batch_size,
    epochs,
    num_gpu,
    dump_path,
):
    model = MeshDataModel(model_name=model_name, embed_dim=embed_dim)
    if checkpoint:
        print(f'==> restore from: {checkpoint}')
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt)

    train_da = DocumentArray.load_binary(train_dataset)
    eval_da = DocumentArray.load_binary(eval_dataset) if eval_dataset else None

    def configure_optimizer(model):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import MultiStepLR

        optimizer = Adam(model.parameters(), lr=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)

        return optimizer, scheduler

    tuned_model = finetuner.fit(
        model,
        train_da,
        eval_data=eval_da,
        preprocess_fn=partial(preprocess, num_points=1024, data_aut=True),
        epochs=epochs,
        batch_size=batch_size,
        loss=TripletLoss(
            miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
        ),
        configure_optimizer=configure_optimizer,
        num_items_per_class=8,
        learning_rate=5e-4,
        device='cuda' if num_gpu > 0 else 'cpu',
    )

    torch.save(
        tuned_model.state_dict(),
        dump_path if dump_path else f'finetuned_{point_model}.pth',
    )


if __name__ == '__main__':
    main()
