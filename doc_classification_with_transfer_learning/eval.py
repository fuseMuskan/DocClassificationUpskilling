import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from torcheval.metrics import MulticlassConfusionMatrix
from sklearn.metrics import classification_report
from datetime import datetime

class_names = ["citizenship", "license", "others", "passport"]


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


def compute_classification_report(input, target, model_name):
    # Convert tensors to NumPy arrays
    y_true = target.numpy()
    y_pred = input.numpy()

    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    # Generating Classification Report
    print("----Generating Classification Report---------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"logs/classification_report_{model_name}_{timestamp}.txt"

    # Print and save the report to the log file
    with open(log_file, "w") as f:
        f.write(
            f"-------------------{model_name}Classification Report-------------------\n"
        )
        f.write(report)
        f.write("-" * 80)

    print("Classification report saved in:", log_file)
