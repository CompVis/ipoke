import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

class Conv2dTransposeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="elu",
        use_bias=True,
        activation_first=False,
        snorm=False
    ):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_channels=norm_dim,num_groups=16)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "elu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if snorm:
            self.conv = spectral_norm(nn.ConvTranspose2d(in_dim, out_dim, ks, st, bias=self.use_bias, padding=padding, output_padding=padding))
        else:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, ks, st, bias=self.use_bias, padding=padding,output_padding=padding)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(x)
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, adain_params):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            adain_params["weight"],
            adain_params["bias"],
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm="in",
        activation="elu",
        pad_type="zero",
        upsampling=False,
        stride = 1,
        snorm=False
    ):
        super(ResBlock, self).__init__()
        self.norm = norm
        self.model = nn.ModuleList()

        if upsampling:
            self.conv1 = Conv2dTransposeBlock(
                dim_in,
                dim_out,
                3,
                2,
                1,
                norm=self.norm,
                activation=activation,
                snorm= snorm
            )

            self.conv2 = Conv2dBlock(
                dim_out,
                dim_out,
                3,
                1,
                1,
                norm=self.norm,
                activation="none",
                pad_type=pad_type,
                snorm=snorm
            )
        else:
            self.conv1 = Conv2dBlock(
                dim_in,
                dim_out,
                3,
                stride,
                1,
                norm=self.norm,
                activation=activation,
                pad_type=pad_type,
                snorm=snorm
            )

            self.conv2 = Conv2dBlock(
                dim_out,
                dim_out,
                3,
                1,
                1,
                norm=self.norm,
                activation="none",
                pad_type=pad_type,
                snorm=snorm
            )

        self.convolve_res = dim_in != dim_out or upsampling or stride != 1
        if self.convolve_res:
            if not upsampling:
                self.res_conv = Conv2dBlock(dim_in,dim_out,3,stride,1,
                                        norm="in",
                                        activation=activation,
                                        pad_type=pad_type,
                                        snorm=snorm)
            else:
                self.res_conv = Conv2dTransposeBlock(dim_in,dim_out,3,2,1,
                                        norm="in",
                                        activation=activation,
                                        snorm=snorm)


    def forward(self, x,adain_params=None):
        residual = x
        if self.convolve_res:
            residual = self.res_conv(residual)
        out = self.conv1(x,adain_params)
        out = self.conv2(out,adain_params)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="elu",
        pad_type="zero",
        use_bias=True,
        activation_first=False,
        snorm=False
    ):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_channels=norm_dim,num_groups=16)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if snorm:
            self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x

class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5*self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std*torch.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input.dtype)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

class AdaINLinear(nn.Module):
    def     __init__(self, in_units, target_units, use_bias=True, actfn=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_units, 2 * target_units, bias=use_bias)
        self.act_fn = actfn()

    def forward(self, x):
        out = self.act_fn(self.linear(x))
        out = {
            "weight": out[:, : out.size(1) // 2],
            "bias": out[:, out.size(1) // 2 :],
        }
        return out

class ADAIN2d(nn.Module):
    def __init__(self, z_dim, in_dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.in_z = AdaINLinear(z_dim,in_dim)
        self.register_buffer("running_mean", torch.zeros(in_dim))
        self.register_buffer("running_var", torch.ones(in_dim))
        self.num_features = in_dim

    def forward(self, x, adain_params):
        adain_params = self.in_z(adain_params)
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            adain_params["weight"],
            adain_params["bias"],
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

class Spade(nn.Module):
    def __init__(self, num_features, dic, num_groups=16):
        super().__init__()
        name = dic['norm']
        self.num_features = num_features
        while self.num_features % num_groups != 0:
            num_groups -= 1
        if name == 'BN' or name == 'batch':
            self.norm = nn.BatchNorm3d(num_features, affine=False, track_running_stats=dic['running_stats'])
        elif name == 'group' or name == 'Group':
            self.norm = nn.GroupNorm(num_groups, num_features, affine=False)
        elif name == 'instance':
            self.norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=dic['running_stats'])
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

        self.conv       = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv_gamma = nn.Conv2d(128, num_features, 3, 1, 1)
        self.conv_beta  = nn.Conv2d(128, num_features, 3, 1, 1)
        self.activate   = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        normalized = self.norm(x)
        y = F.interpolate(y, mode='bilinear', size=x.shape[-2:], align_corners=True).cuda()
        y = self.activate(self.conv(y))
        gamma = self.conv_gamma(y)#.unsqueeze(2).repeat_interleave(x.size(2), 2)
        beta  = self.conv_beta(y)#.unsqueeze(2).repeat_interleave(x.size(2), 2)
        return normalized * (1 + gamma) + beta


class Norm3D(nn.Module):
    def __init__(self, num_features, dic, num_groups=16):
        super().__init__()
        name = dic['norm']
        self.num_features = num_features
        if name == 'BN' or name == 'batch':
            self.bn = nn.BatchNorm3d(num_features, affine=True, track_running_stats=dic['running_stats'])
        elif name == 'group' or name == 'Group':
            self.bn = nn.GroupNorm(num_groups, num_features, affine=True)
        elif name == 'instance':
            self.bn = nn.InstanceNorm3d(num_features, affine=True, track_running_stats=dic['running_stats'])
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

    def forward(self, x, y=None):
        out = self.bn(x)
        return out


class ADAIN(nn.Module):
    def __init__(self, num_features, dic, num_groups=16):
        super().__init__()
        name = 'instance'# dic['norm']
        self.num_features = num_features
        while self.num_features % num_groups != 0:
            num_groups -= 1
        if name == 'BN' or name == 'batch':
            self.bn = nn.BatchNorm3d(num_features, affine=False, track_running_stats=dic['running_stats'])
        elif name == 'group' or name == 'Group':
            self.bn = nn.GroupNorm(num_groups, num_features, affine=False)
        elif name == 'instance':
            self.bn = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=dic['running_stats'])
        elif name == 'layer':
            self.bn = nn.LayerNorm(num_features, elementwise_affine=False)
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

        self.linear = nn.Linear(dic['z_dim'], num_features*2)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.linear(y).chunk(2, 1)
        out = (1 + gamma.view(-1, self.num_features, 1, 1, 1)) * out + beta.view(-1, self.num_features, 1, 1, 1)
        return out