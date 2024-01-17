# Introduction
Fine tune different model for document classification.
Following models were fine-tuned under different hyperparameters and the experiment was tracked using tensorboard:
* ResNet
* Inception v3
* Densenet
* VGG11_bn
* Squeezenet
* Alexnet


# DATASET
Dataset was collected and downloaded from google.
It can be accessed here: <a href = "https://drive.google.com/drive/folders/1MawKiWPK_0ZAaHWZbgQMsOc23id6n2UF?usp=sharing">DatasetLink </a>


Note: The dataloader expects the following directory structure:
- data/
  - documents/
    - train/
      - citizenship/
        - image01.jpeg
        - ...
      - license/
      - passport/
      - others/
    - test/
      - citizenship/
      - license/
      - passport/
      - others/



