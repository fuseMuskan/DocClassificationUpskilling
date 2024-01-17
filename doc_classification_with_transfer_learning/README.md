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

# USAGE
You can use the following notebook to fine tune the models: <a href = "https://github.com/fuseMuskan/DocClassificationUpskilling/blob/main/doc_classification_with_transfer_learning/document_classification_with_transfer_learning.ipynb"> Notebook </a>

# REFERENCES
* https://pytorch.org/vision/stable/transforms.html#v2-api-ref
* https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
* https://pytorch.org/tutorials/beginner/data_loading_tutorial.html



