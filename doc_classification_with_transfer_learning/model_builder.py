"""
Pytorch model code to instantiate a custom Model
"""
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import data_setup
import pathlib


def train_test_dataloader(input_size, train_dir, test_dir):
    custom_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(75),
            transforms.RandomHorizontalFlip(0.6),
            transforms.RandomVerticalFlip(0.6),
            transforms.RandomPerspective(0.4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("[INFO] Preparing Data Loaders")
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=custom_transforms,
        batch_size=4,
    )
    return train_dataloader, test_dataloader, class_names


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    train_dir,
    test_dir,
    use_pretrained=True,
):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    print("[INFO] Initializing Model")

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        train_dataloader, test_dataloader, class_names = train_test_dataloader(
            input_size, train_dir, test_dir
        )

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        train_dataloader, test_dataloader, class_names = train_test_dataloader(
            input_size, train_dir, test_dir
        )

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        train_dataloader, test_dataloader, class_names = train_test_dataloader(
            input_size, train_dir, test_dir
        )

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

        train_dataloader, test_dataloader, class_names = train_test_dataloader(
            input_size, train_dir, test_dir
        )

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        train_dataloader, test_dataloader, class_names = train_test_dataloader(
            input_size, train_dir, test_dir
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size, train_dataloader, test_dataloader, class_names
