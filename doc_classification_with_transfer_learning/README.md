# Introduction
Fine tune different model for document classification.
Following models were fine-tuned under different hyperparameters and the experiment was tracked using tensorboard:
* ResNet
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
    - validation/
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
cd ..
pip install -r requirements.txt
```

## FINE TUNING MODEL

* MODEL = name of the model to choose ["resnet", "alexnet", "vgg", "squeeznet", "densenet"]
* EPOCHS = num of epochs
* BATCH_SIZE = batch size
* DATA_DIR = directory that contains the dataset
* LEARNING_RATE = learning rate used in the optimizer
* OUTPUT_MODEL = the name of the model to be saved after training
* USE_CLASS_WEIGHTS = (True of False) if True uses the class weights else doesnot use the class weights

```
python train.py --MODEL="resnet" --EPOCHS=5 --BATCH_SIZE=1 --DATA_DIR=F:\ApprecentishipProgram\DocumentClassificationUp\DocClassificationUpskilling\dataset --OUTPUT_MODEL=model_1 --USE_CLASS_WEIGHTS=True
```

## RUNNING INFERENCE

```
python doc_classify.py --model_path=path/to_your_model/model_name.onnx --image_path=path/to_the_document_image/image.jpg
```

## LOGS
Different evaluation metrics like classification report, confusion matrix and loss were tracked on tensorboard and can be accesed here:-<br>
* <a href = "https://drive.google.com/drive/folders/1UgaCPBt3jjixTcxc3Nk4i6Q6puc7TSWp?usp=sharing">Runs (for locally running tensorboard)</a>
* <a href = "https://drive.google.com/drive/folders/1nr9RNolX5_D6TNolKoOuW8NFuTz1bZdb?usp=sharing">Classification Report</a>

# REFERENCES
* https://pytorch.org/vision/stable/transforms.html#v2-api-ref
* https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
* https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
* https://thenewstack.io/tutorial-using-a-pre-trained-onnx-model-for-inferencing/
