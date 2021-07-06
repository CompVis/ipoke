import torch
import torch.nn as nn
from torch.distributions import Normal


class FlowLoss(nn.Module):
    def __init__(self,spatial_mean=False, logdet_weight=1.):
        super().__init__()
        # self.config = config
        self.spatial_mean = spatial_mean
        self.logdet_weight = logdet_weight

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample, spatial_mean=self.spatial_mean))
        assert len(logdet.shape) == 1
        if self.spatial_mean:
            h,w = sample.shape[-2:]
            nlogdet_loss = -torch.mean(logdet) / (h*w)
        else:
            nlogdet_loss = -torch.mean(logdet)

        loss = nll_loss + self.logdet_weight*nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample),spatial_mean=self.spatial_mean))
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss": nll_loss,
            'logdet_weight': self.logdet_weight
        }
        return loss, log

class FlowLossAlternative(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config

    def forward(self, sample, logdet):
        nll_loss = torch.mean(torch.sum(0.5*torch.pow(sample, 2), dim=1))
        nlogdet_loss = - logdet.mean()


        loss = nll_loss + nlogdet_loss
        reference_sample = torch.randn_like(sample)
        reference_nll_loss = torch.mean(torch.sum(0.5*torch.pow(reference_sample, 2), dim=1))
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss": nll_loss
        }
        return loss, log

class ExtendedFlowLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.config = config

    def forward(self, sample_x, sample_v, logdet):
        nll_loss_x = torch.mean(nll(sample_x))
        nll_loss_v = torch.mean(nll(sample_v))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss_x + nll_loss_v + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample_x)))
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss_x": nll_loss_x,
            "nll_loss_v": nll_loss_v
        }
        return loss, log

def nll(sample, spatial_mean= False):
    if spatial_mean:
        return 0.5 * torch.sum(torch.mean(torch.pow(sample, 2),dim=[2,3]), dim=1)
    else:
        return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])


class GaussianLogP(nn.Module):

    def __init__(self,mu=0.,sigma=1.):
        super().__init__()
        self.dist = Normal(loc=mu,scale=sigma)

    def forward(self,sample,logdet):
        nll_log_loss = torch.sum(self.dist.log_prob(sample)) / sample.size(0)
        nlogdet_loss = torch.mean(logdet)
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        nll_loss = torch.mean(nll(sample))
        loss = - (nll_log_loss + nlogdet_loss)
        log = {"flow_loss":loss,
               "reference_nll_loss":reference_nll_loss,
               "nlogdet_loss":-nlogdet_loss,
               "nll_loss": nll_loss,
               "nll_log_loss":-nll_log_loss}

        return loss, log