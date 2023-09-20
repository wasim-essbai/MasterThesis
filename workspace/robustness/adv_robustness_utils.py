import torch
import torchbnn as bnn
import numpy as np
import torch.nn.functional as F

def get_aleatoric(p_hat, device):
    mean_pred = torch.mean(p_hat, axis=0)
    pred_value = None

    aleat_mat = torch.zeros(10, 10).to(device)
    for i in range(p_hat.shape[0]):
        aleat_mat += torch.diag(p_hat[i]) - torch.outer(p_hat[i], p_hat[i])
    return torch.mean(torch.diag(aleat_mat / p_hat.shape[0]))


def get_epistemic_unc(p_hat, device):
    mean_pred = torch.mean(p_hat, axis=0)

    epis_mat = torch.zeros(10, 10).to(device)
    for i in range(p_hat.shape[0]):
        epis_mat += torch.outer((p_hat[i] - mean_pred), (p_hat[i] - mean_pred))
    return torch.mean(torch.diag(epis_mat / p_hat.shape[0]))

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon, cf ):
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    # Metrics counter    
    correct = 0
    unknown = 0
    adv_examples = []
    aleatoric_sum = 0
    epistemic_sum = 0
    progress = 0
    dataset_len = len(test_loader)
    # Loop over all examples in test set
    for data, target in test_loader:
        print('\r' + ' Eps: ' + str(epsilon) + ' Evaluation: ' + str(round(100 * progress / dataset_len, 2)) + '% complete..', end="")
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        target_one_hot = F.one_hot(target, num_classes=10)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        p_hat_list = []
        loss = 0
        for i in range(10):
          dist_pred = model(data.view(data.shape[0], -1))
          p_hat_list.append(dist_pred.mean.squeeze())

          # Calculate the loss
          kl = kl_loss(model)
          nll = torch.mean(-dist_pred.log_prob(target_one_hot.float()), dim=0).mean()
          loss_i = nll + kl/target.shape[0]
          loss += loss_i

        p_hat = torch.stack(p_hat_list)
        loss = loss/10

        output = cf(p_hat, 0.8)
        init_pred = output # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            progress += 1
            continue

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        #data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        # Reapply normalization
        #perturbed_data_normalized = perturbed_data - torch.min(perturbed_data)

        # Re-classify the perturbed image
        p_hat_list = []
        loss = 0
        for i in range(10):
          dist_pred = model(perturbed_data.view(perturbed_data.shape[0], -1))
          p_hat_list.append(dist_pred.mean.squeeze())
        p_hat = torch.stack(p_hat_list)
        
        output = cf(p_hat, 0.8)

        # Check for success
        final_pred = output # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if final_pred.item() == -1:
              unknown += 1
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        aleatoric_sum += get_aleatoric(p_hat, device)
        epistemic_sum += get_epistemic_unc(p_hat, device)
        progress += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(dataset_len - unknown)
    final_unkn = unknown/float(dataset_len)
    aleatoric = aleatoric_sum / float(dataset_len)
    epistemic = epistemic_sum / float(dataset_len)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {dataset_len} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, final_unkn, aleatoric, epistemic, adv_examples