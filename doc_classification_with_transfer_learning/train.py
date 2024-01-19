"""
Trains a PyTorch image classification model using device-agnostic code
"""
import os
import argparse
import torch
import data_setup, engine, model_builder, utils
from pathlib import Path
from torchvision import transforms
from utils import create_writer

# Extracting argparse values

parser = argparse.ArgumentParser()

parser.add_argument("--MODEL", type=str, help="Model name [Alexnet, Resnet,...]")
parser.add_argument("--EPOCHS", type=int, help="Number of epochs for training")
parser.add_argument("--BATCH_SIZE", type=int, help="Batch size for training")
parser.add_argument("--DATA_DIR", type=str, help="Path to the data directory")
parser.add_argument("--OUTPUT_MODEL", type=str, help="Name of the model to be saved")


args = parser.parse_args()

# Set up Hyperparameters
MODEL_NAME = args.MODEL
EPOCHS = args.EPOCHS
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = 0.001
# LEARNING_RATE = args.LEARNING_RATE
DATA_DIR = args.DATA_DIR
OUTPUT_MODEL = args.OUTPUT_MODEL


# Setup directories
data_path = Path(DATA_DIR)
train_dir = data_path / "train"
test_dir = data_path / "test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


(
    model,
    input_size,
    train_dataloader,
    test_dataloader,
    class_names,
) = model_builder.initialize_model(
    model_name=MODEL_NAME,
    num_classes=4,
    feature_extract=True,
    train_dir=train_dir,
    test_dir=test_dir,
)


# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create summary writer to track experiment
writer = create_writer("doc_classification", MODEL_NAME, f"{EPOCHS} epochs")


# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    writer=writer,
    device=device,
)

# Save the model with help from utils.py
utils.save_model(model=model, target_dir="models", model_name=f"{OUTPUT_MODEL}.pth")
