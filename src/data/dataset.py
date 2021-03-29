import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from typing import Tuple, Iterable
from sklearn.feature_extraction.text import TfidfVectorizer


class CommentDataset(Dataset):
    def __init__(self, csv_name):
        df = pd.read_csv(csv_name)

        self.labels = sorted(df.y.unique().tolist())
        X = TfidfVectorizer().fit_transform(df.x.values)
        y = df.y.apply(lambda x: self.labels.index(x)).values

        self.X = torch.from_numpy(X.toarray()).float().to_sparse()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self):
        fmt_str = ["Comment dataset for sentiment analysis."]
        fmt_str.append(f"Number of comments: {self.__len__()}")
        fmt_str.append(f"Number of words: {self.X.shape[1]}")
        fmt_str.append(f"Labels: {self.labels}")
        return '\n'.join(fmt_str)


def split_dataset(ds: Dataset,
                  train_size: float = 0.8) -> Tuple[Subset, Subset]:
    """Function taking an image folder and splitting it in a train and test set.
    To achieve random results, we recommend to set up the seed for
    reproductibility with `torch.manual_seed(seed)`.
    The splitting ratio can be the number of samples (int) or the dataset
    proportion (flaot between 0 and 1).
    Args:
        ds (Dataset): The dataset to be split
        train_size (float, optional): The number of samples or the
        proportion of the dataset. Defaults to 0.9.
    Returns:
        Tuple[Subset, Subset]: The training and validation subsets.
    """

    if isinstance(train_size, float):
        train_size = int(len(ds)*train_size)
    train_ds, val_ds = random_split(ds, [train_size, len(ds)-train_size])
    return train_ds, val_ds


def get_dataloaders(ds: Dataset,
                    bs: Iterable,
                    train_size: float = 0.8
                    ) -> Tuple[DataLoader, DataLoader]:
    train_bs, val_bs = bs
    train_ds, val_ds = split_dataset(ds, train_size=train_size)
    train_dl = DataLoader(train_ds, batch_size=train_bs)
    val_dl = DataLoader(val_ds, batch_size=val_bs)
    return train_dl, val_dl
