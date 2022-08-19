import pathlib

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from datasets import ModelNet40
from executor import MeshDataEncoderPL


@click.command()
@click.option('--train_dataset', help='The training dataset file path')
@click.option(
    '--split_ratio',
    default=0.8,
    help='The proportion of training samples out of the whole training dataset',
)
@click.option('--eval_dataset', help='The evaluation dataset file path')
@click.option('--embed_dim', default=512, help='The embedding dimension')
@click.option(
    '--checkpoint_path',
    type=click.Path(file_okay=True, path_type=pathlib.Path),
    help='The path of checkpoint',
)
@click.option('--model_name', default='pointnet', help='The model name')
@click.option('--batch_size', default=128, help='The size of each batch')
@click.option('--epochs', default=50, help='The epochs of training process')
@click.option('--use-gpu/--no-use-gpu', default=True, help='If True to use gpu')
@click.option(
    '--interactive', default=False, help='set to True if you have unlabeled data'
)
@click.option(
    '--devices', default=7, help='The number of gpus/tpus you can use for training'
)
@click.option('--seed', default=10, help='The random seed for reproducing results')
def main(
    train_dataset,
    split_ratio,
    eval_dataset,
    model_name,
    embed_dim,
    batch_size,
    epochs,
    use_gpu,
    checkpoint_path,
    interactive,
    devices,
    seed,
):
    torch.manual_seed(seed)
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    if checkpoint_path:
        model = MeshDataEncoderPL.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
    else:
        model = MeshDataEncoderPL(
            default_model_name=model_name, embed_dim=embed_dim, device=device
        )

    train_and_val_data = ModelNet40(train_dataset, seed=seed)
    tot_len = len(train_and_val_data)
    train_len = int(tot_len * split_ratio)
    validate_len = tot_len - train_len
    train_data, validate_data = random_split(
        train_and_val_data, [train_len, validate_len]
    )
    test_data = ModelNet40(eval_dataset, seed=seed)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    validate_loader = DataLoader(
        validate_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    logger = TensorBoardLogger(
        save_dir='./logs',
        log_graph=True,
        name='{}_dim_{}_batch_{}_epochs_{}'.format(
            model_name, embed_dim, batch_size, epochs
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='val_loss',
        mode='min',
        filename='{epoch:02d}-{val_loss:.2f}',
    )

    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    model.train()
    trainer.fit(model, train_loader, validate_loader)
    print(checkpoint_callback.best_model_path)

    model.eval()
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
