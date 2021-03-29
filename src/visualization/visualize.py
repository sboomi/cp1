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
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Normalized confusion matrix (%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if fn:
        fig.savefig(fn)
        plt.close()
