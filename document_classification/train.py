"""
Trains a PyTorch image classification model using device-agnostic code
"""
import os
import argparse
import torch
import data_setup, engine, model_builder, utils
from pathlib import Path

from torchvision import transforms

# Extracting argparse values

parser = argparse.ArgumentParser()

parser.add_argument("--EPOCHS", type=int, help="Number of epochs for training")
parser.add_argument("--BATCH_SIZE", type=int, help="Batch size for training")
parser.add_argument(
    "--HIDDEN_UNITS", type=int, help="Number of hidden units in the model"
)
parser.add_argument("--LEARNING_RATE", type=float, help="Learning rate for optimizer")
parser.add_argument("--DATA_DIR", type=str, help="Path to the data directory")
parser.add_argument("--OUTPUT_MODEL", type=str, help="Name of the model to be saved")


args = parser.parse_args()

# Set up Hyperparameters
EPOCHS = args.EPOCHS
BATCH_SIZE = args.BATCH_SIZE
HIDDEN_UNITS = args.HIDDEN_UNITS
LEARNING_RATE = args.LEARNING_RATE
DATA_DIR = args.DATA_DIR
OUTPUT_MODEL = args.OUTPUT_MODEL


# Setup directories
data_path = Path(DATA_DIR)
train_dir = data_path / "train"
test_dir = data_path / "test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
)

# Create model with help from model_builder.py
model = model_builder.CustomModel(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    device=device,
)

# Save the model with help from utils.py
utils.save_model(model=model, target_dir="models", model_name=f"{OUTPUT_MODEL}.pth")
