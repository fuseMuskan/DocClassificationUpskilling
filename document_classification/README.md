# DATASET

Dataset was collected and downloaded from google.
It can be accessed here: <a href = "https://drive.google.com/drive/folders/1MawKiWPK_0ZAaHWZbgQMsOc23id6n2UF?usp=sharing">DatasetLink </a>



# USAGE

* EPOCHS = num of epochs
* BATCH_SIZE = batch size
* DATA_DIR = directory that contains the dataset
* LEARNING_RATE = set learning rate
* HIDDEN_UNITS = num of hidden units
* OUTPUT_MODEL = the name of the model to be saved after training


```
!python train.py --EPOCHS=5 --BATCH_SIZE=1 --DATA_DIR=/content/drive/MyDrive/DocumentUpskilling/dataset/documents --LEARNING_RATE=0.001 --HIDDEN_UNITS=10 --OUTPUT_MODEL=model_0_1
```
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



