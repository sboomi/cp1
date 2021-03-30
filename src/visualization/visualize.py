import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels="auto", fn=None):
    name = fn.stem
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(f"CM for {name}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(100 * (cm / cm.sum(axis=1)),
                annot=True, ax=ax, fmt='.2f',
                vmax=100, vmin=0, center=50,
                xticklabels=labels, yticklabels=labels,
                cmap='Blues')
    ax.set_title("Normalized confusion matrix (%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if fn:
        fig.savefig(fn)
        plt.close()


def plot_cm_nn(res_dict, labels="auto", fn=None):
    last_train_cm = res_dict["train"]["cm"][-1]
    last_val_cm = res_dict["val"]["cm"][-1]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    sns.heatmap(last_train_cm / last_train_cm.sum(axis=1),
                ax=ax1, annot=True, yticklabels=labels,
                xticklabels=labels, cmap='Blues')
    sns.heatmap(last_val_cm / last_val_cm.sum(axis=1),
                ax=ax2, annot=True, yticklabels=labels,
                xticklabels=labels, cmap='Blues')

    ax1.set_title("Train set")
    ax2.set_title("Val set")

    ax1.set_ylabel("Actual")
    ax2.set_ylabel("Actual")

    ax1.set_xlabel("Predicted")
    ax2.set_xlabel("Predicted")
    if fn:
        fig.savefig(fn)
        plt.close()


def plot_loss_acc(res_dict, fn=None):
    train_loss = res_dict["train"]["loss"]
    val_loss = res_dict["val"]["loss"]

    train_acc = [rep_ep["accuracy"] for rep_ep in res_dict["train"]["report"]]
    val_acc = [rep_ep["accuracy"] for rep_ep in res_dict["val"]["report"]]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.plot(train_loss, label="Loss (train)")
    ax1.plot(val_loss, label="Loss (val)")
    ax1.set_ylabel("Loss value")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_acc, label="Accuracy (train)")
    ax2.plot(val_acc, label="Accuracy (val)")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    if fn:
        fig.savefig(fn)
        plt.close()
