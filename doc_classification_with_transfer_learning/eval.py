import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from torcheval.metrics import MulticlassConfusionMatrix


def calculate_confusion_matrix(input, target):
    metric = MulticlassConfusionMatrix(4)
    metric.update(input, target)
    conf_matrix = metric.compute()
    return conf_matrix


def plot_confusion_matrix(conf_matrix, class_names):
    fig, ax = plt.subplots(figsize=(len(class_names), len(class_names)))
    sns.set(font_scale=1.2)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return fig


def save_confusion_matrix(conf_matrix):
    class_names = ["citizenship", "license", "others", "passport"]

    fig = plot_confusion_matrix(
        conf_matrix=conf_matrix.numpy(), class_names=class_names
    )
    return fig
