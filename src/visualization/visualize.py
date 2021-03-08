import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, fn=None):
    name = fn.stem
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(f"CM for {name}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(np.round(100 * cm / cm.sum()), annot=True, ax=ax, fmt='.2f')
    fig.savefig(fn)
    