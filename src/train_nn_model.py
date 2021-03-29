import click
import logging
import time
import datetime as dt
import coloredlogs
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple
from data.dataset import CommentDataset, get_dataloaders
from models.nn_model import SimpleNN, train_model


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option("--batch-size", "-bs", "bs",
              nargs=2, type=click.Tuple([int, int]))
@click.option("--epochs", "-ep", "n_epochs", type=int)
@click.option("--compare-models", "-cm", "comp_model",
              type=click.Path(exists=True))
@click.option("--random-seed", "-rs", "seed", type=int, default=32451365)
def main(csv_path: str,
         model_path: str,
         comp_model: str,
         bs: Tuple[int, int],
         n_epochs: int,
         seed: int
         ) -> None:
    start_time = time.time()
    logger = logging.getLogger('ml-model')
    logger.info("Begin training on Deep learning model")
    torch.manual_seed(seed)

    model_path = Path(model_path)

    ds = CommentDataset(csv_path)
    logger.info(f"Loaded dataset of {len(ds)} samples.")
    logger.info(f"Dataset labels: {ds.labels}")
    n_words = ds[0][0].shape[0]
    logger.info(f"N° of words: {n_words}")

    logger.info(f"Loading dataloaders with batch sizes {bs}")
    train_dl, val_dl = get_dataloaders(ds, bs, train_size=0.8)

    logger.info(f"Loading NN model with input {n_words}, output 2")
    model = SimpleNN(n_words, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model, criterion = model.to(device), criterion.to(device)

    results = train_model(train_dl, val_dl, n_epochs, model,
                          criterion, optimizer, device, ds.labels)

    torch.save(model.state_dict(), model_path / "nn_model.pt")

    with open(model_path / 'nn_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    tot_time = str(dt.timedelta(seconds=end_time-start_time))
    h, mn, s = tot_time.split(":")
    logger.info(f"Script ended in {h}h, {mn}min and {s}s")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    coloredlogs.install()

    main()
