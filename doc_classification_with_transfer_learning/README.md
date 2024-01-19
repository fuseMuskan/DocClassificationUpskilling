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

* You can use the following notebook to fine tune the models: <a href = "https://github.com/fuseMuskan/DocClassificationUpskilling/blob/main/doc_classification_with_transfer_learning/document_classification_with_transfer_learning.ipynb"> Notebook </a>

* To save the model, we can use <a href = "https://github.com/fuseMuskan/DocClassificationUpskilling/blob/main/doc_classification_with_transfer_learning/document_classification.ipynb">Notebook</a>


## INSTALLING REQUIREMENTS

```
pip install -r requirements.txt
```

## FINE TUNING MODEL

* MODEL = name of the model to choose ["resnet", "alexnet", "vgg", "squeeznet", "densenet"]
* EPOCHS = num of epochs
* BATCH_SIZE = batch size
* DATA_DIR = directory that contains the dataset
* LEARNING_RATE =
* OUTPUT_MODEL = the name of the model to be saved after training

```
python train.py --MODEL="resnet" --EPOCHS=5 --BATCH_SIZE=1 --DATA_DIR=F:\ApprecentishipProgram\DocumentClassificationUp\DocClassificationUpskilling\dataset --OUTPUT_MODEL=model_1
```

## RUNNING INFERENCE

```
python doc_classify.py --model_path=path/to_your_model/model_name.onnx --image_path=path/to_the_document_image/image.jpg
```

# REFERENCES
* https://pytorch.org/vision/stable/transforms.html#v2-api-ref
* https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
* https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
* https://thenewstack.io/tutorial-using-a-pre-trained-onnx-model-for-inferencing/
