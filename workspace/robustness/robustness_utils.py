import numpy as np
import torch
import os
from torchvision import transforms
import sys
from torch.distributions.one_hot_categorical import OneHotCategorical

sys.path.append('./workspace/data/mnist_data')
from cmnist_dataset import CMNISTDataset


def evaluate_bnn(model, test_loader, classification_function, conf_level=0.5):
    with torch.no_grad():
        testCorrect = 0
        testUnknown = 0
        aleatoric_sum = -1
        for data, target in test_loader:
            for j in range(0, data.shape[0]):
                img = data[j]
                y = target[j]
                p_hat_list = []
                for i in range(15):
                    dist_pred = model(img.view(1, -1))
                    p_hat_list.append(dist_pred.mean.squeeze())
                p_hat = torch.stack(p_hat_list)
                pred_values = classification_function(p_hat, conf_level)
                testCorrect += torch.sum(pred_values == y)
                testUnknown += torch.sum(pred_values == -1)
        accuracy = np.round(testCorrect * 100 / (len(test_loader.dataset) - testUnknown), 2)
        unknown_ration = np.round(testUnknown * 100 / len(test_loader.dataset), 2)
        aleatoric = aleatoric_sum / len(test_loader.dataset)
        return accuracy, unknown_ration, aleatoric


def evaluate_ann(model, test_loader):
    with torch.no_grad():
        datasetLength = len(test_loader)
        testCorrect = 0
        level = 0

        for data, target in test_loader:
            pred = model(data.view(data.shape[0], -1))

            pred_values = torch.max(pred, 1).indices
            testCorrect += torch.sum(pred_values == target)

        return np.round(testCorrect * 100 / len(test_loader.dataset), 2)


def evaluate_alteration(model, alteration_name, is_bnn=True, classification_function=None):
    base_path = f'/content/drive/MyDrive/MasterThesis/workspace/mnist_alt/{alteration_name}'

    dir_list = next(os.walk(base_path))[1]
    accuracy_list = []
    unknown_ratio_list = []
    aleatoric_list = []
    step_list = []
    level = 0
    for step_dir in dir_list:
        test_loader = torch.utils.data.DataLoader(
            CMNISTDataset(root_dir=base_path + '/' + step_dir + '/',
                          train=False,
                          labels_root=base_path + '/',
                          transform=transforms.ToTensor()),
            batch_size=128, shuffle=False)
        if is_bnn:
            accuracy, unknown_ratio, aleatoric = evaluate_bnn(model, test_loader, classification_function)
            accuracy_list.append(accuracy)
            unknown_ratio_list.append(unknown_ratio)
            aleatoric_list.append(aleatoric)
        else:
            accuracy_list.append(evaluate_ann(model, test_loader))
        step_list.append(float(step_dir))
        level += 1
        print('\r' + ' Evaluation: ' + str(round(100 * level / len(dir_list), 2)) + '% complete..', end="")
    return accuracy_list, step_list, unknown_ratio_list, aleatoric_list
