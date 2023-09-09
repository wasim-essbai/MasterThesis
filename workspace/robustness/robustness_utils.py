import numpy as np
import torch
import os
from torchvision import transforms
import sys
from torch.distributions.one_hot_categorical import OneHotCategorical
from scipy import integrate

sys.path.append('./workspace/data/mnist_data')
from cmnist_dataset import CMNISTDataset
from functions import linear_tolerance, linear_dist, uniform_dist
from result_eval import ResultEval

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_aleatoric(p_hat):
    mean_pred = torch.mean(p_hat, axis=0)
    pred_value = None

    aleat_mat = torch.zeros(10, 10).to(device)
    for i in range(p_hat.shape[0]):
        aleat_mat += torch.diag(p_hat[i]) - torch.outer(p_hat[i], p_hat[i])
    return torch.mean(torch.diag(aleat_mat / p_hat.shape[0]))


def get_epistemic_unc(p_hat):
    mean_pred = torch.mean(p_hat, axis=0)

    epis_mat = torch.zeros(10, 10).to(device)
    for i in range(p_hat.shape[0]):
        epis_mat += torch.outer((p_hat[i] - mean_pred), (p_hat[i] - mean_pred))
    return torch.mean(torch.diag(epis_mat / p_hat.shape[0]))


def get_robustness(y, x, maxAcc, th):
    ua = np.max(x)
    la = np.min(x)

    y_tol = linear_tolerance(y, maxAcc, th, True)
    y_int = y_tol * uniform_dist(x, ua, la)
    return integrate.trapezoid(y_int, x) / 2 + 0.5


def get_robustness_ind(y, x, maxAcc, th):
    ua = np.max(x)
    la = np.min(x)

    y_tol = linear_tolerance(y, maxAcc, th, False)
    y_int = y_tol * uniform_dist(x, ua, la)
    return integrate.trapezoid(y_int, x) / 2 + 0.5


def evaluate_bnn(model, test_loader, classification_functions, conf_level=0.8):
    with torch.no_grad():
        datasetLength = len(test_loader.dataset)
        testCorrect = torch.zeros(len(classification_functions))
        testUnknown = torch.zeros(len(classification_functions))
        aleatoric_sum = 0
        epistemic_sum = 0
        for data_hat, target_hat in test_loader:
            (data, target) = (data_hat.to(device), target_hat.to(device))
            for j in range(0, data.shape[0]):
                img = data[j]
                y = target[j]
                p_hat_list = []
                for i in range(10):
                    dist_pred = model(img.view(1, -1))
                    p_hat_list.append(dist_pred.mean.squeeze())
                p_hat = torch.stack(p_hat_list)

                pred_values = []
                for cf in classification_functions:
                    pred_values.append(cf(p_hat, conf_level))
                for i in range(len(pred_values)):
                    testCorrect[i] += torch.sum(pred_values[i] == y.cpu())
                    testUnknown[i] += torch.sum(pred_values[i] == -1)
                aleatoric_sum += get_aleatoric(p_hat)
                epistemic_sum += get_epistemic_unc(p_hat)
        accuracy = []
        unknown_ratio = []
        for i in range(len(classification_functions)):
            accuracy.append(torch.round(testCorrect[i] * 100 / (datasetLength - testUnknown[i]), decimals=2))
            unknown_ratio.append(torch.round(testUnknown[i] * 100 / datasetLength, decimals=2))
        aleatoric = aleatoric_sum / datasetLength
        epistemic = epistemic_sum / datasetLength
        return accuracy, unknown_ratio, aleatoric, epistemic


def evaluate_ann(model, test_loader):
    with torch.no_grad():
        datasetLength = len(test_loader)
        testCorrect = 0
        level = 0

        for data_hat, target_hat in test_loader:
            (data, target) = (data_hat.to(device), target_hat.to(device))
            pred = model(data.view(data.shape[0], -1))

            pred_values = torch.max(pred, 1).indices
            testCorrect += torch.sum(pred_values == target)

        return np.round(testCorrect * 100 / len(test_loader.dataset), 2)


def evaluate_alteration(model, alteration_name, is_bnn=True, classification_functions=None):
    base_path = f'/content/drive/MyDrive/MasterThesis/workspace/mnist_alt/{alteration_name}'

    dir_list = next(os.walk(base_path))[1]
    result_evaluation = None
    if classification_functions is not None:
        result_evaluation = []
        for cf in classification_functions:
            result_evaluation = ResultEval(cf.__name__)
    else:
        result_evaluation = [ResultEval(None)]

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
            accuracy, unknown_ratio, aleatoric, epistemic = evaluate_bnn(model, test_loader, classification_functions)
            for i in range(len(accuracy)):
                result_evaluation[i].accuracy.append(accuracy[i].cpu())
                result_evaluation[i].unkn.append(unknown_ratio[i].cpu())
                result_evaluation[i].aleatoric.append(aleatoric.cpu())
                result_evaluation[i].epistemic.append(epistemic.cpu())
        else:
            accuracy = evaluate_ann(model, test_loader)
            result_evaluation.accuracy.append(accuracy.cpu())
            #accuracy_list.append(evaluate_ann(model, test_loader))
            #result_evaluation
        step_list.append(float(step_dir))
        level += 1

        print('\r' + ' Evaluation: ' + str(round(100 * level / len(dir_list), 2)) + '% complete..', end="")
    for re in result_evaluation:
        re.steps = step_list
    return result_evaluation
