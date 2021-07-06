import torch
import torch.nn.functional as F


def KLDLoss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))


def calculate_adaptive_weight(nll_loss, g_loss, discriminator_weight, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, epoch, threshold=0, value=0.):
    if epoch < threshold:
        weight = value
    return weight
