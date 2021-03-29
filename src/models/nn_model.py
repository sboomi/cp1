import logging
import coloredlogs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm, trange


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
coloredlogs.install()
logger = logging.getLogger('nn-training')


class SimpleNN(nn.Module):
    def __init__(self, in_values, out_values):
        super().__init__()
        self.dense1 = nn.Linear(in_values, 12673)
        self.drop1 = nn.Dropout()
        self.dense2 = nn.Linear(12673, 4000)
        self.drop2 = nn.Dropout()
        self.dense3 = nn.Linear(4000, 500)
        self.drop3 = nn.Dropout()
        self.last_dense = nn.Linear(500, out_values)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        x = F.relu(self.dense2(x))
        x = self.drop2(x)
        x = F.relu(self.dense3(x))
        x = self.drop3(x)
        x = self.last_dense(x)
        return x


@torch.no_grad()
def evaluate(dl, model, criterion, device, total_it=-1):
    model.eval()

    eval_loss = 0.0
    y_true = []
    y_preds = []

    max_it = len(dl) if total_it == -1 else total_it
    for bn, (X, y) in tqdm(enumerate(dl), unit="batch",
                           desc=f"Model evaluation (bs={dl.batch_size})"):
        if bn == max_it:
            break
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        eval_loss += loss.detach().item()
        y_true.append(y.detach().cpu().numpy())
        y_preds.append(y_pred.argmax(dim=1).cpu().numpy())
    return eval_loss / max_it, np.block(y_true), np.block(y_preds)


def train_per_epoch(train_dl, model, criterion, optimizer, device):
    model.train()
    for X, y in tqdm(train_dl, unit="batch",
                     desc="Model fitting"):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()


def train_model(train_dl, val_dl, n_epochs, model,
                criterion, optimizer, device, labels):
    res_train = {"train": {"loss": [], "cm": [], "report": []},
                 "val": {"loss": [], "cm": [], "report": []}}
    for epoch in trange(n_epochs, unit="epoch"):
        train_per_epoch(train_dl, model, criterion, optimizer, device)

        tot_it = len(val_dl)*val_dl.batch_size // train_dl.batch_size
        train_loss, train_true, train_pred = evaluate(train_dl,
                                                      model,
                                                      criterion,
                                                      device,
                                                      total_it=tot_it)

        val_loss, val_true, val_pred = evaluate(val_dl,
                                                model,
                                                criterion,
                                                device)

        logger.info(f"EPOCH {epoch}")
        logger.info(f"Train loss: {train_loss} / Val loss: {val_loss}")
        res_train["train"]["loss"].append(train_loss)
        res_train["val"]["loss"].append(val_loss)

        res_train["train"]["cm"].append(confusion_matrix(train_true,
                                                         train_pred))
        res_train["val"]["cm"].append(confusion_matrix(val_true,
                                                       val_pred))

        train_report = classification_report(train_true,
                                             train_pred,
                                             target_names=labels,
                                             output_dict=True)
        val_report = classification_report(val_true,
                                           val_pred,
                                           target_names=labels,
                                           output_dict=True)
        res_train["train"]["report"].append(train_report)
        res_train["val"]["report"].append(val_report)
        logger.info(f"Train Accuracy: {train_report['accuracy']} / "
                    f"Val accuracy: {val_report['accuracy']}")

    return res_train
