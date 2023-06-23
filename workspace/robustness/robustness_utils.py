import numpy as np
import torch
import os
from torchvision import transforms
import sys

sys.path.append('./workspace/data/mnist_data')
from cmnist_dataset import CMNISTDataset


def evaluate(model, test_loader):
    with torch.no_grad():
        datasetLength = len(test_loader)
        testCorrect = 0
        level = 0

        for data, target in test_loader:
            dist_pred = model(data.view(data.shape[0], -1))

            pred_values = torch.max(dist_pred.mean, 1).indices
            testCorrect += torch.sum(pred_values == target)

        return np.round(testCorrect * 100 / len(test_loader.dataset), 2)


def evaluate_alteration(model, alteration_name):
    base_path = f'/content/drive/MyDrive/MasterThesis/workspace/mnist_alt/{alteration_name}'

    accuracy_list = []
    step_list = []
    for step_dir in os.listdir(base_path):
        test_loader = torch.utils.data.DataLoader(
            CMNISTDataset(root_dir=base_path + '/' + step_dir + '/',
                          train=False,
                          labels_root=base_path + '/',
                          transform=transforms.ToTensor()),
            batch_size=128, shuffle=False)
        accuracy_list.append(evaluate(model, test_loader))
        step_list.append(float(step_dir))

    return accuracy_list, step_list
