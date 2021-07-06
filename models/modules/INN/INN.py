"""Vanilla Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle"""
import torch, os
import torch.nn as nn
from models.modules.INN.flow_blocks import (
    ConditionalFlow,
    UnconditionalFlow,
    UnconditionalFlow2,
    UnconditionalMixCDFConvFlow,
    UnconditionalMaCow,
    UnconditionalMaCowFlow,
    UnconditionalExtendedLeapFlow,
    UnconditionalLeapFlow,
    ConditionalConvFlow
)
from models.modules.INN.modules import BasicFullyConnectedNet
from models.modules.INN.macow2 import MaCowStep,MultiScaleInternal, MultiscaleStack, MultiscaleMixCDF,HierarchicalConvCouplingFlow


class SupervisedTransformer(nn.Module):
    """Vanilla version. No multiscale support."""

    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.config = config
        in_channels = config["flow_in_channels"]
        mid_channels = config["flow_mid_channels_factor"] * config["flow_in_channels"]
        hidden_depth = config["flow_hidden_depth"]
        n_flows = config["n_flows"]
        conditioning_option = config["flow_conditioning_option"]
        embedding_channels = (
            config["flow_embedding_channels"]
            if "flow_embedding_channels" in config
            else config["flow_in_channels"]
        )
        # self.num_class_channels = (
        #     config["flow_num_classes"]
        #     if "flow_num_classes"
        #     else config["flow_in_channels"]
        # )

        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalFlow(
            in_channels=in_channels,
            embedding_dim=self.emb_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            conditioning_option=conditioning_option,
        )

        # self.pretrain = config['pretrain']
        # model_path =  config["first_stage_path"]
        # import sys
        # sys.path.insert(0, model_path + 'code/')

    def sample(self, shape, label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, label)
        return sample

    def forward(self, input, conditioning, reverse=False, train=False):

        # if self.pretrain:
        #     with torch.no_grad():
        #         embed_img = self.embedder(label[0])[1].reshape(input.size(0), -1).detach()
        # else:
        #     embed_img = self.embedder(label[0]).reshape(input.size(0), -1)

        # cond = torch.cat(conditioning, dim=1)
        # embedding = torch.cat((embed_speed, embed_dir), dim=1)

        if reverse:
            return self.reverse(input, conditioning)

        out, logdet = self.flow(input, conditioning)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, label):
        return self.flow(out, label, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight")

class SupervisedConvTransformer(nn.Module):
    """Vanilla version. No multiscale support."""

    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.config = config
        in_channels = config["flow_in_channels"]
        mid_channels = config["flow_mid_channels_factor"] * config["flow_in_channels"]
        hidden_depth = config["n_blocks"]
        n_flows = config["n_flows"]
        embedding_channels = (
            config["h_channels"]
            if "h_channels" in config
            else config["flow_in_channels"]
        )
        # self.num_class_channels = (
        #     config["flow_num_classes"]
        #     if "flow_num_classes"
        #     else config["flow_in_channels"]
        # )

        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalConvFlow(
            in_channels=in_channels,
            embedding_dim=self.emb_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
        )

        # self.pretrain = config['pretrain']
        # model_path =  config["first_stage_path"]
        # import sys
        # sys.path.insert(0, model_path + 'code/')

    def sample(self, shape, label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, label)
        return sample

    def forward(self, input, conditioning, reverse=False, train=False):

        # if self.pretrain:
        #     with torch.no_grad():
        #         embed_img = self.embedder(label[0])[1].reshape(input.size(0), -1).detach()
        # else:
        #     embed_img = self.embedder(label[0]).reshape(input.size(0), -1)

        # cond = torch.cat(conditioning, dim=1)
        # embedding = torch.cat((embed_speed, embed_dir), dim=1)

        if reverse:
            return self.reverse(input, conditioning)

        out, logdet = self.flow(input, conditioning)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, label):
        return self.flow(out, label, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight")


class UnsupervisedTransformer(nn.Module):
    """Vanilla version. No multiscale support."""

    def __init__(self, **kwargs):
        super().__init__()
        # self.config = config

        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]

        self.in_channels = in_channels

        self.flow = UnconditionalFlow(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
        )

    # def sample(self, shape):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     z_tilde = torch.randn(shape).to(device)
    #     sample = self.reverse(z_tilde)
    #     return sample

    def forward(self, input, reverse=False, train=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out):
        return self.flow(out, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )


class UnsupervisedTransformer2(nn.Module):
    """To support uneven dims and get rid of leaky relu thing"""

    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]

        self.in_channels = in_channels

        self.flow = UnconditionalFlow2(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
        )

    def sample(self, shape, device="cpu"):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample.squeeze(dim=-1).squeeze(dim=-1)

    def forward(self, input, reverse=False, train=False):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out):
        if len(out.shape) == 2:
            out = out[:, :, None, None]
        return self.flow(out, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )

class UnsupervisedTransformer3(nn.Module):
    """Data depended initialization"""

    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]

        self.in_channels = in_channels

        self.flow = UnconditionalFlow2(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            data_init=True
        )

    def sample(self, shape, device="cpu"):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample.squeeze(dim=-1).squeeze(dim=-1)

    def forward(self, input, reverse=False, train=False):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out):
        if len(out.shape) == 2:
            out = out[:, :, None, None]
        return self.flow(out, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )


class UnsupervisedConvTransformer(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config
        self.flow = UnconditionalMixCDFConvFlow(self.config)

    def forward(self,input,reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class UnsupervisedMaCowTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.flow = UnconditionalMaCow(self.config)

    def forward(self,input,reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class UnsupervisedMaCowTransformer2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.flow = UnconditionalMaCowFlow(self.config)

    def forward(self,input,reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class UnsupervisedMaCowTransformer3(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.flow = MultiScaleInternal(MaCowStep,num_steps=self.config["num_steps"],in_channels=self.config["flow_in_channels"],
                                       hidden_channels=self.config["flow_mid_channels"],h_channels=0,
                                       factor=self.config["factor"],transform=self.config["transform"],
                                       prior_transform=self.config["prior_transform"],kernel_size=self.config["kernel_size"],
                                       coupling_type=self.config["coupling_type"],activation=self.config["activation"],
                                       use_1x1=self.config["use1x1"] if "use1x1" in self.config else False)

    def forward(self,input,reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class UnsupervisedHierarchicalMixCDFTransformer(nn.Module):


    def __init__(self, config):
        super().__init__()

        self.config = config
        self.flow = MultiscaleMixCDF(num_steps=self.config['num_steps'],dimension=self.config['latent_dim'],
                                     hidden_channels_factor=self.config['flow_mid_channels_factor'],h_channels=0,
                                     factor=self.config['factor'],heads=self.config['flow_attn_heads'],
                                     components=self.config['flow_cdf_components'],
                                     coupling_type=self.config["coupling_type"], activation=self.config["activation"],
                                     use_1x1=self.config["use1x1"] if "use1x1" in self.config else False
                                     )


    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet


    def reverse(self, out):
        return self.flow(out, reverse=True)


    def sample(self, shape, device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class MacowTransformerMultiStep(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.flow = MultiscaleStack(self.config)

    def forward(self, input, xc=None, reverse=False):
        if reverse:
            return self.reverse(input, xc)
        out, logdet = self.flow(input, xc)
        return out, logdet

    def reverse(self, out, xc=None):
        return self.flow(out, xc, reverse=True)

    def sample(self, shape,xc=None, device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde,xc)
        return sample


class SupervisedMacowTransformer(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        condition_nice = 'condition_nice' in self.config and self.config['condition_nice']
        attention = 'attention' in self.config and self.config['attention']
        heads = self.config['flow_attn_heads']
        ssize = 'ssize' in self.config and self.config['ssize']
        cond_conv = 'cond_conv' in config and config['cond_conv']
        cond_conv_hidden_channels = config['cond_conv_hidden_channels'] if 'cond_conv_hidden_channels' else None
        p_drop = config['p_dropout'] if 'p_dropout' in config else 0.
        self.flow = MultiScaleInternal(MaCowStep, num_steps=self.config["num_steps"], in_channels=self.config["flow_in_channels"],
                                       hidden_channels=self.config["flow_mid_channels"], h_channels=self.config["h_channels"],
                                       factor=self.config["factor"], transform=self.config["transform"],
                                       prior_transform=self.config["prior_transform"], kernel_size=self.config["kernel_size"],
                                       coupling_type=self.config["coupling_type"], activation=self.config["activation"],
                                       use_1x1=self.config["use1x1"] if "use1x1" in self.config else False,
                                       condition_nice=condition_nice,attention=attention,heads=heads,spatial_size=ssize,
                                       cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                       p_dropout=p_drop
                                       )

    def forward(self, input, cond, reverse=False):
        if reverse:
            return self.reverse(input,cond)
        out, logdet = self.flow(input,cond)
        return out, logdet

    def reverse(self, out, cond):
        return self.flow(out, cond, reverse=True)

    def sample(self, shape,cond, device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde,cond)
        return sample


class SupervisedHierarchicalCouplingTransformer(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        # attention = 'attention' in self.config and self.config['attention']


        assert self.config['h_channels'] > 0
        self.flow = HierarchicalConvCouplingFlow(num_steps=self.config['num_steps'],in_channels=self.config['flow_in_channels'],
                                                 hidden_channels_factor=self.config['flow_mid_channels_factor'],
                                                 h_channels=self.config['h_channels'],factor=self.config['factor'],
                                                 transform=self.config['transform'],prior_transform=self.config['prior_transform'],
                                                 activation=self.config['activation'],condition_nice=True,
                                                 attention=self.config['attention'],heads=self.config['flow_attn_heads'],spatial_size=self.config['ssize'],
                                                 n_blocks=self.config['n_blocks'])

    def forward(self, input, cond, reverse=False):
        if reverse:
            return self.reverse(input, cond)
        out, logdet = self.flow(input, cond)
        return out, logdet

    def reverse(self, out, cond):
        return self.flow(out, cond, reverse=True)

    def sample(self, shape, cond, device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, cond)
        return sample

class UnsupervisedExtendedLeapFrogTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.config = config

        in_channels = config["flow_in_channels"]
        mid_channels = config["flow_mid_channels"]
        hidden_depth = config["flow_hidden_depth"]
        n_flows = config["n_flows"]
        delta_t = config['delta_t'] if 'delta_t' in config else 1.

        self.in_channels = in_channels

        self.flow = UnconditionalExtendedLeapFlow(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            delta_t=delta_t
        )

    # def sample(self, shape):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     z_tilde = torch.randn(shape).to(device)
    #     sample = self.reverse(z_tilde)
    #     return sample

    def forward(self, input, v, reverse=False):
        if v.dim() > 2:
            v = v.squeeze(-1).squeeze(-1)
        if reverse:
            return self.reverse(input,v)
        out, v, logdet = self.flow(input,v)
        return out, v, logdet

    def reverse(self, out,v):
        return self.flow(out,v, reverse=True)


class UnsupervisedLeapFrogTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.config = config

        in_channels = config["flow_in_channels"]
        mid_channels = config["flow_mid_channels"]
        hidden_depth = config["flow_hidden_depth"]
        n_flows = config["n_flows"]
        delta_t = config['delta_t'] if 'delta_t' in config else 1.

        self.in_channels = in_channels

        self.flow = UnconditionalLeapFlow(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            delta_t=delta_t
        )

    # def sample(self, shape):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     z_tilde = torch.randn(shape).to(device)
    #     sample = self.reverse(z_tilde)
    #     return sample

    def forward(self, input, v, reverse=False):
        if v.dim() > 2:
            v = v.squeeze(-1).squeeze(-1)
        if reverse:
            return self.reverse(input,v)
        out, v, logdet = self.flow(input,v)
        return out, v, logdet

    def reverse(self, out,v):
        return self.flow(out,v, reverse=True)


