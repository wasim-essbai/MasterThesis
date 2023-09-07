import numpy as np
import torch
import os
from torchvision import transforms
import sys
from torch.distributions.one_hot_categorical import OneHotCategorical
from scipy import integrate

sys.path.append('./workspace/data/mnist_data')
from cmnist_dataset import CMNISTDataset
from functions import linear_tolerance, linear_dist


def get_aleatoric(p_hat):
    mean_pred = torch.mean(p_hat, axis=0)
    pred_value = None

    aleat_mat = torch.zeros(10, 10)
    for i in range(p_hat.shape[0]):
        aleat_mat += torch.diag(p_hat[i]) - torch.outer(p_hat[i], p_hat[i])
    return torch.mean(torch.diag(aleat_mat / p_hat.shape[0]))


def get_epistemic_unc(p_hat):
    mean_pred = torch.mean(p_hat, axis=0)

    epis_mat = torch.zeros(10, 10)
    for i in range(p_hat.shape[0]):
        epis_mat += torch.outer((p_hat[i] - mean_pred), (p_hat[i] - mean_pred))
    return torch.mean(torch.diag(epis_mat / p_hat.shape[0]))


def get_robustness(y, x, maxAcc, th):
    ua = np.max(x)
    la = np.min(x)

    y_tol = linear_tolerance(y, maxAcc, th, True)
    y_int = y_tol * linear_dist(x, ua, la)
    return integrate.trapezoid(y_int, x) / 2 + 0.5


def get_robustness_ind(y, x, maxAcc, th):
    ua = np.max(x)
    la = np.min(x)

    y_tol = linear_tolerance(y, maxAcc, th, False)
    y_int = y_tol * linear_dist(x, ua, la)
    return integrate.trapezoid(y_int, x) / 2 + 0.5


def evaluate_bnn(model, test_loader, classification_function, conf_level=0.5):
    with torch.no_grad():
        datasetLength = len(test_loader.dataset)
        testCorrect = 0
        testUnknown = 0
        aleatoric_sum = 0
        epistemic_sum = 0
        for data, target in test_loader:
            for j in range(0, data.shape[0]):
                img = data[j]
                y = target[j]
                p_hat_list = []
                for i in range(10):
                    dist_pred = model(img.view(1, -1))
                    p_hat_list.append(dist_pred.mean.squeeze())
                p_hat = torch.stack(p_hat_list)
                pred_values = classification_function(p_hat, conf_level)
                testCorrect += torch.sum(pred_values == y)
                testUnknown += torch.sum(pred_values == -1)
                aleatoric_sum += get_aleatoric(p_hat)
                epistemic_sum += get_epistemic_unc(p_hat)
        accuracy = np.round(testCorrect * 100 / (datasetLength - testUnknown), 2)
        unknown_ration = np.round(testUnknown * 100 / datasetLength, 2)
        aleatoric = aleatoric_sum / datasetLength
        return accuracy, unknown_ration, aleatoric, epistemic_sum


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
    epistemic_list = []
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
            accuracy, unknown_ratio, aleatoric, epistemic = evaluate_bnn(model, test_loader, classification_function)
            accuracy_list.append(accuracy)
            unknown_ratio_list.append(unknown_ratio)
            aleatoric_list.append(aleatoric)
            epistemic_list.append(epistemic)
        else:
            accuracy_list.append(evaluate_ann(model, test_loader))
        step_list.append(float(step_dir))
        level += 1
        print('\r' + ' Evaluation: ' + str(round(100 * level / len(dir_list), 2)) + '% complete..', end="")
    return accuracy_list, step_list, unknown_ratio_list, aleatoric_list, epistemic_list
