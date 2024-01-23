import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(train_dataloader, class_names):
    class_counts = {class_name: 0 for class_name in class_names}
    for _, targets in train_dataloader:
        for target in targets:
            class_name = class_names[target]
            class_counts[class_name] += 1
    # Calculate class weights
    total_samples = sum(class_counts.values())
    class_weights = [
        total_samples / count for class_name, count in class_counts.items()
    ]
    class_weights = torch.FloatTensor(class_weights)
    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


def create_focal_loss_criterion(train_dataloader, class_names, gamma):
    class_weights = calculate_class_weights(train_dataloader, class_names)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    return criterion
