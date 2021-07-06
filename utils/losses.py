import torch
from torch import nn

import torchvision

class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # X = self.normalize(X)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def normalize(self, x):
        x = x.permute(1, 0, 2, 3)
        for i in range(3):
            x[i] = x[i] * self.std[i] + self.mean[i]
        return x.permute(1, 0, 2, 3)

def KL(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))

def kl_conv(mu,logvar):
    mu = mu.reshape(mu.size(0),-1)
    logvar = logvar.reshape(logvar.size(0),-1)

    var = torch.exp(logvar)

    return torch.mean(0.5 * torch.sum(torch.pow(mu, 2) + var - 1.0 - logvar, dim=-1))

def fmap_loss(fmap1, fmap2, loss):
    recp_loss = 0
    for idx in range(len(fmap1)):
        if loss == 'l1':
            recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
        if loss == 'l2':
            recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
    return recp_loss / len(fmap1)

class VGGLoss(nn.Module):
    def __init__(self, weighted=False):
        super(VGGLoss, self).__init__()
        self.vgg = VGG().cuda()
        self.weighted = weighted
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        fmap1, fmap2 = self.vgg(x), self.vgg(y)
        if self.weighted:
            recp_loss = 0
            for idx in range(len(fmap1)):
                recp_loss += self.weights[idx] * self.criterion(fmap2[idx], fmap1[idx])
            return recp_loss
        else:
            return fmap_loss(fmap1, fmap2, loss='l1')