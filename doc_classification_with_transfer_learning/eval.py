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


def plot_to_image(fig):
    # Saving the plot to a PNG in memory.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    # Converting PNG buffer to PIL Image
    pil_image = Image.open(buffer).convert("RGBA")

    # Converting PIL Image to PyTorch tensor
    image = torch.tensor(np.array(pil_image)).permute(2, 0, 1).float() / 255.0

    # Adding the batch dimension
    image = image.unsqueeze(0)

    return image


def save_confusion_matrix(conf_matrix):
    class_names = ["citizenship", "license", "others", "passport"]

    fig = plot_confusion_matrix(
        conf_matrix=conf_matrix.numpy(), class_names=class_names
    )
    # conf_image = plot_to_image(fig)
    print("PASSED..............")
    return fig


input = torch.tensor([0, 2, 1, 3])
target = torch.tensor([0, 1, 2, 3])

conf_matrix = calculate_confusion_matrix(input, target)
save_confusion_matrix(conf_matrix)
