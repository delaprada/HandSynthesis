import torch
from torch.nn import functional as F


def mse(pred, gt, weight=None):
    assert pred.shape == gt.shape, "shapes of pred and get must be equal in MSE Loss"
    if weight is None:
        loss = F.mse_loss(pred, gt)
    else:
        loss = torch.mean(weight * ((pred - gt) ** 2))

    return loss


def loss_pck_fn(pred, gt, thresh):
    """
    # Calculate the PCK loss with radius.
    :param pred:
    :param gt:
    :param thresh:
    :return:
    """
    dist = torch.sqrt(torch.sum(((gt - pred) ** 2), dim=2))  # BS x n_joints x 3
    loss_pck = torch.sum(dist <= thresh)
    loss_pck = loss_pck.float()
    # Normalise by the number of joints. Its NOT normalised by batch size
    loss_pck /= gt.size(1)
    return loss_pck


def kl_div(mu, logvar, dim_joints, per_neuron=False, weights=None, n_joints=21):
    """
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014, https://arxiv.org/abs/1312.6114
    -KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param mu:
    :param logvar:
    :param dim_joints:
    :param per_neuron:
    :param weights:
    :return:
    """
    if per_neuron:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0, keepdim=True)
    else:
        if weights is None:
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            KLD = -0.5 * torch.sum(weights * (1 + logvar - mu.pow(2) - logvar.exp()))
        # Normalise by same number of elements as in reconstruction
        batch_size = mu.size(0)
        KLD /= batch_size * n_joints * dim_joints
    return KLD
