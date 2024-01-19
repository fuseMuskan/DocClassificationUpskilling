from torcheval.metrics import MulticlassConfusionMatrix


def calculate_confusion_matrix(input, target):
    metric = MulticlassConfusionMatrix(4)
    metric.update(input, target)
    conf_matrix = metric.compute()
    return conf_matrix
