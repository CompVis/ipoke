import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import scipy.linalg as alg
from torch.optim.optimizer import Optimizer

version_higher = (torch.__version__ >= "1.5.0")


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data,
                                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                    p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                    p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data,
                                                                memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,
                                                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data,
                                                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                            p.data)

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                            1.0 - beta2 ** state['step'])

                    if state['rho_t'] > 4:  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)

        return loss


from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


# define networks
class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


# append normalization layer
class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, x):
        return x


# Universal norm layer
class NormLayer2d(nn.Module):
    def __init__(self, nf, norm):
        super(NormLayer2d, self).__init__()

        if norm == "batch":
            self.norm = nn.BatchNorm2d(nf, affine=True, track_running_stats=True)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(nf, affine=True, track_running_stats=False)
        elif norm == "batch-instance":
            self.norm = BatchInstanceNorm2d(nf, affine=True)
        elif norm == "layer":
            self.norm = nn.LayerNorm(nf)
        else:
            self.norm = DummyLayer()

    def forward(self, x):
        return self.norm(x)


class SwishActivation(nn.Module):
    def __init__(self):
        super(SwishActivation, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MishActivation(nn.Module):
    def __init__(self):
        super(MishActivation, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, bias=False, norm="batch", act="mish", use_spec=True,
                 transpose=False, output_padding=0, shuffle=False):
        super(ConvBlock2d, self).__init__()

        if transpose and not shuffle:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias,
                                           output_padding=output_padding)
        elif transpose and shuffle:
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=k, stride=s, padding=p, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias)

        if use_spec:
            self.conv = nn.utils.spectral_norm(self.conv)

        nn.init.normal_(self.conv.weight, 0., 0.05)

        rest = []
        if shuffle:
            rest += [nn.PixelShuffle(2)]

        rest += [NormLayer2d(out_channels, norm)]
        if act == "lrelu":
            rest += [nn.LeakyReLU(0.2, True)]
        elif act == "relu":
            rest += [nn.ReLU(True)]
        elif act == "tanh":
            rest += [nn.Tanh()]
        elif act == "hardtanh":
            rest += [nn.Hardtanh()]
        elif act == "swish":
            rest += [SwishActivation()]
        elif act == "mish":
            rest += [MishActivation()]

        self.rest = nn.Sequential(*rest)

    def forward(self, x):
        x = self.conv(x)
        x = self.rest(x)

        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, use_spec=True):
        super(Linear, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        if use_spec:
            self.lin = nn.utils.spectral_norm(self.lin)

    def forward(self, x):
        return self.lin(x)


# simple layer for printing debug information
class DebugLayer(nn.Module):
    def __init__(self):
        super(DebugLayer, self).__init__()

    def forward(self, input):
        print(input.shape)
        # print(input)
        return input


class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // r, channels, kernel_size=1, stride=1, padding=0)

        self.act1 = nn.ReLU()

    def forward(self, x):
        # do global average pooling
        pool = torch.mean(x, dim=(2, 3), keepdim=True)
        a = self.conv1(pool)
        a = self.act1(a)
        a = self.conv2(a)
        return x * a.expand_as(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()

        self.convf = ConvBlock2d(channels, channels // 8, k=1, s=1, p=0, act="none", norm="none")
        self.convg = ConvBlock2d(channels, channels // 8, k=1, s=1, p=0, act="none", norm="none")
        self.convh = ConvBlock2d(channels, channels // 8, k=1, s=1, p=0, act="none", norm="none")
        self.convv = ConvBlock2d(channels // 8, channels, k=1, s=1, p=0, act="none", norm="none")
        self.act = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f = self.convf(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)  # transpose
        g = self.convg(x).view(x.size(0), -1, x.size(2) * x.size(3))
        attn = self.act(torch.bmm(f, g))  # B x HW x HW
        h = self.convh(x).view(x.size(0), -1, x.size(2) * x.size(3))
        attn = torch.bmm(h, attn.permute(0, 2, 1))
        o = self.convv(attn.view(x.size(0), -1, x.size(2), x.size(3)))

        return x + self.gamma * o


class Encoder(nn.Module):
    def __init__(self, in_channels, ngf=64, nz=16, max_filters=128, norm="batch", variational=False,
                 attn=False):
        super(Encoder, self).__init__()

        self.nz = nz
        self.variational = variational

        encoder = [nn.ReflectionPad2d(1), ConvBlock2d(in_channels, ngf, k=3, s=1, p=0, norm=norm),
                   ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        channels = []
        for i in range(2):
            in_channels = ngf
            ngf = min(ngf * 2, max_filters)
            encoder += [ConvBlock2d(in_channels, ngf, k=4, s=2, p=1, norm=norm)]
            channels.append(in_channels)
        encoder += [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        if not self.variational:
            encoder += [ConvBlock2d(ngf, self.nz, k=1, s=1, p=0, norm="none", act="none")]
        self.encoder = nn.Sequential(*encoder)

        if self.variational:
            self.mu = ConvBlock2d(ngf, self.nz, k=1, s=1, p=0, norm="none", act="none")
            self.logvar = ConvBlock2d(ngf, self.nz, k=1, s=1, p=0, norm="none", act="none")

    def forward(self, x):
        x = self.encoder(x)

        if self.variational:
            mu = self.mu(x)
            logvar = self.logvar(x)
            eps = torch.randn_like(logvar)
            z = mu + torch.exp(logvar * 0.5) * eps
            return mu, logvar, z
        else:
            return x

    def inference(self, x):
        x = self.encoder(x)

        if self.variational:
            mu = self.mu(x)
            logvar = self.logvar(x)
            eps = torch.randn_like(logvar)
            z = mu + torch.exp(logvar * 0.5) * eps
            return z
        else:
            return x

    def set_grad(self, requires):
        for param in self.parameters():
            param.requires_grad = requires


class Decoder(nn.Module):
    def __init__(self, ngf=32, nz=16, max_filters=128, norm="batch", dual=False):
        super(Decoder, self).__init__()

        self.dual = dual

        channels = []
        for i in range(2):
            in_channels = ngf
            ngf = min(ngf * 2, max_filters)
            channels.append(in_channels)

        decoder = [ConvBlock2d(nz, ngf, k=1, s=1, p=0, norm=norm, transpose=True)]
        decoder += [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm, transpose=True)]
        for i in reversed(range(2)):
            in_channels = ngf
            ngf = channels[i]
            # decoder += [ConvBlock2d(in_channels, ngf, k=4, s=2, p=1, norm=norm, transpose=True)]
            decoder += [nn.UpsamplingNearest2d(scale_factor=2.0),
                        ConvBlock2d(in_channels, ngf, k=3, s=1, p=1, norm=norm, transpose=True)]
        decoder += [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm, transpose=True)]
        if not dual:
            decoder += [ConvBlock2d(ngf, 1, k=3, s=1, p=1, act="relu", norm="none", transpose=True)]
        self.decoder = nn.Sequential(*decoder)

        # design head
        if dual:
            self.head1 = ConvBlock2d(ngf, 1, k=5, s=1, p=2, act="tanh", norm="none")

            self.head2 = ConvBlock2d(ngf, 1, k=5, s=1, p=2, act="none", norm="none")

    def forward(self, x):
        x = self.decoder(x)
        if self.dual:
            return self.head1(x), self.head2(x)
        else:
            return x

    def inference(self, x):
        x = self.decoder(x)
        if self.dual:
            return self.head1(x), torch.sigmoid(self.head2(x))
        else:
            return x

    def set_grad(self, requires):
        for param in self.parameters():
            param.requires_grad = requires


class Discriminator(nn.Module):  # receptive field=23
    def __init__(self, in_channels, ngf=32, max_filters=128, norm="none"):
        super(Discriminator, self).__init__()

        encoder = [nn.ReflectionPad2d(1), ConvBlock2d(in_channels, ngf, k=3, s=1, p=0, norm=norm),
                   ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        channels = []
        for i in range(2):
            in_channels = ngf
            ngf = min(ngf * 2, max_filters)
            encoder += [ConvBlock2d(in_channels, ngf, k=4, s=2, p=1, norm=norm)]
            channels.append(in_channels)
        self.encoder = nn.Sequential(*encoder)
        pred = [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm),
                ConvBlock2d(ngf, 1, k=1, s=1, p=0, norm="none", act="none")]
        self.pred = nn.Sequential(*pred)

    def forward(self, x):
        feats = self.encoder(x)
        return self.pred(feats)

    def infer(self, x):
        feats = self.encoder(x)
        return self.pred(feats)

    def get_feats(self, x):
        return self.encoder(x)

    def set_grad(self, requires):
        for param in self.parameters():
            param.requires_grad = requires


class MI_Discriminator1(nn.Module):  # receptive field=23
    def __init__(self, in_channels, ngf=32, max_filters=128, norm="none"):
        super(MI_Discriminator1, self).__init__()

        encoder = [ConvBlock2d(in_channels, ngf, k=5, s=1, p=2, norm=norm),
                   ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        channels = []
        for i in range(2):
            in_channels = ngf
            ngf = min(ngf * 2, max_filters)
            encoder += [ConvBlock2d(in_channels, ngf, k=3, s=2, p=1, norm=norm)]
            channels.append(in_channels)
        encoder += [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        encoder += [ConvBlock2d(ngf, ngf, k=3, s=1, p=1, norm=norm)]
        self.encoder = nn.Sequential(*encoder)

        self.pred = ConvBlock2d(ngf, 1, k=1, s=1, p=0, norm="none", act="none")

    def forward(self, x):
        feats = self.encoder(x)
        return self.pred(feats)

    def infer(self, x):
        feats = self.encoder(x)
        return self.pred(feats)

    def set_grad(self, requires):
        for param in self.parameters():
            param.requires_grad = requires


class MI_Discriminator2(nn.Module):
    def __init__(self, in_channels=18, ndf=64, max_filters=128, norm="none"):
        super(MI_Discriminator2, self).__init__()

        block = [ConvBlock2d(in_channels, ndf, k=1, s=1, p=0, norm=norm),
                 ConvBlock2d(ndf, 2 * ndf, k=1, s=1, p=0, norm=norm),
                 ConvBlock2d(2 * ndf, 1, k=1, s=1, p=0, norm="none", act="none")]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    def infer(self, x):
        return self.block(x)

    def set_grad(self, requires):
        for param in self.parameters():
            param.requires_grad = requires


class InvertibleDummyLayer(nn.Module):
    def __init__(self):
        super(InvertibleDummyLayer, self).__init__()

    def logdet(self):
        return 0.0

    def forward(self, x):
        return x

    def reverse(self, x):
        return x


class InvertibleLeakyReLU(nn.Module):
    def __init__(self, alpha=0.9):
        super(InvertibleLeakyReLU, self).__init__()

        self.alpha = alpha

    def logdet(self):
        return 0.0

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha, inplace=False)

    def reverse(self, x):
        nmask = (x <= 0).to(x.device)
        pmask = (x > 0).to(x.device) * self.alpha

        return x / (nmask + pmask)


class InvertibleActivationLayer(nn.Module):
    def __init__(self, type="lrelu", alpha=1.0):
        super(InvertibleActivationLayer, self).__init__()

        if type == "lrelu":
            self.act = InvertibleLeakyReLU(alpha)
        elif type == "none":
            self.act = InvertibleDummyLayer()
        else:
            print("Activation function {} does not exist!".format(type))
            self.act = InvertibleDummyLayer()

    def logdet(self):
        return self.act.logdet()

    def forward(self, x):
        return self.act(x)

    def reverse(self, x):
        return self.act.reverse(x)


class ActNorm2d(nn.Module):
    def __init__(self, nf):
        super(ActNorm2d, self).__init__()
        self.nf = nf  # number of features

        self.scale = nn.Parameter(torch.ones(1, nf, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, nf, 1, 1))
        self.register_buffer("initialized", torch.zeros(1))

    def initialize(self, x):
        with torch.no_grad():
            x = x.permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
            mean = x.mean(dim=1).view(1, x.size(0), 1, 1)
            std = x.std(dim=1, unbiased=False).view(1, x.size(0), 1, 1)

            assert not torch.sum(torch.isnan(std)), "Nan occured in Actnorm! {} {}".format(std, mean)
            assert not torch.sum(torch.isinf(std)), "Inf occured in Actnorm! {} {}".format(std, mean)

            self.scale.data.copy_((1.0 / std + 1e-6).data)
            self.bias.data.copy_(mean.data)
        self.initialized.fill_(1)

    def logdet(self):
        val = torch.sum(torch.log(torch.abs(self.scale)), dim=1) * torch.ones(self.b, self.h, self.w).to(
            self.scale.device)
        assert not torch.sum(torch.isnan(val)), "Nan occured in Actnorm logdet! {}".format(
            torch.sum(torch.log(torch.abs(self.scale) + 1e-6)))
        assert not torch.sum(torch.isinf(val)), "Inf occured in Actnorm logdet! {}".format(
            torch.sum(torch.log(torch.abs(self.scale) + 1e-6)))
        return val

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        x = (x - self.bias) * self.scale
        self.b = x.size(0)
        self.h = x.size(2)
        self.w = x.size(3)

        return x

    def reverse(self, x):
        x = (x / self.scale) + self.bias

        return x


class InvertibleNormLayer2d(nn.Module):
    def __init__(self, nf, type="act"):
        super(InvertibleNormLayer2d, self).__init__()

        if type == "act":
            self.norm = ActNorm2d(nf)
        elif type == "none":
            self.norm = InvertibleDummyLayer()
        else:
            self.norm = InvertibleDummyLayer()

    def logdet(self):
        return self.norm.logdet()

    def forward(self, x):
        return self.norm(x)

    def reverse(self, x):
        return self.norm.reverse(x)


class ZeroConv(nn.Module):
    def __init__(self, nf, nf_out, k=1, s=1, p=0):
        super(ZeroConv, self).__init__()
        self.conv = nn.Conv2d(nf, nf_out, kernel_size=k, stride=s, padding=p)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class AffineCouplingLayer2d(nn.Module):
    def __init__(self, nf, nf_hidden, k=3, p=1):
        super(AffineCouplingLayer2d, self).__init__()
        self.dima = nf // 2
        self.dimb = nf - self.dima

        s1 = [ConvBlock2d(self.dimb, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ConvBlock2d(nf_hidden, self.dima, k=k, s=1, p=p, norm="none", act="tanh")]
        self.s1 = nn.Sequential(*s1)
        s2 = [ConvBlock2d(self.dima, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ConvBlock2d(nf_hidden, self.dimb, k=k, s=1, p=p, norm="none", act="tanh")]
        self.s2 = nn.Sequential(*s2)

        t1 = [ConvBlock2d(self.dimb, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ConvBlock2d(nf_hidden, self.dima, k=k, s=1, p=p, norm="none", act="none")]
        self.t1 = nn.Sequential(*t1)
        t2 = [ConvBlock2d(self.dima, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ConvBlock2d(nf_hidden, self.dimb, k=k, s=1, p=p, norm="none", act="none")]
        self.t2 = nn.Sequential(*t2)

    def logdet(self):
        val = torch.sum(self.vals1, dim=1) + torch.sum(self.vals2, dim=1)
        assert not torch.sum(torch.isnan(val)), "Nan occured in Coupling logdet {}".format(val)
        assert not torch.sum(torch.isinf(val)), "Inf occured in Coupling logdet {}".format(val)
        return val  # .unsqueeze(1)

    def forward(self, x):
        # first of all split data
        x1 = x[:, :self.dima, :, :]
        x2 = x[:, self.dima:, :, :]

        self.vals1 = self.s1(x2)
        v1 = x1 * self.vals1.exp() + self.t1(x2)
        self.vals2 = self.s2(v1)
        v2 = x2 * self.vals2.exp() + self.t2(v1)

        v = torch.cat([v1, v2], dim=1)

        return v

    def reverse(self, x):
        # first of all split data
        x1 = x[:, :self.dima, :, :]
        x2 = x[:, self.dima:, :, :]

        v2 = (x2 - self.t2(x1)) * self.s2(x1).neg().exp()
        v1 = (x1 - self.t1(v2)) * self.s1(v2).neg().exp()

        v = torch.cat([v1, v2], dim=1)

        return v


class ConditionalAffineCouplingLayer2d(nn.Module):
    def __init__(self, nf, nf_c, nf_hidden, k=3, p=1):
        super(ConditionalAffineCouplingLayer2d, self).__init__()
        self.dima = nf // 2
        self.dimb = nf - self.dima

        s1 = [ConvBlock2d(self.dimb + nf_c, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ZeroConv(nf_hidden, self.dima, k=k, s=1, p=p), nn.Tanh()]
        self.s1 = nn.Sequential(*s1)
        s2 = [ConvBlock2d(self.dima + nf_c, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ZeroConv(nf_hidden, self.dimb, k=k, s=1, p=p), nn.Tanh()]
        self.s2 = nn.Sequential(*s2)

        t1 = [ConvBlock2d(self.dimb + nf_c, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ZeroConv(nf_hidden, self.dima, k=k, s=1, p=p)]
        self.t1 = nn.Sequential(*t1)
        t2 = [ConvBlock2d(self.dima + nf_c, nf_hidden, k=k, s=1, p=p, norm="batch"),
              ZeroConv(nf_hidden, self.dimb, k=k, s=1, p=p)]
        self.t2 = nn.Sequential(*t2)

    def logdet(self):
        val = torch.sum(self.vals1, dim=1) + torch.sum(self.vals2, dim=1)
        assert not torch.sum(torch.isnan(val)), "Nan occured in Coupling logdet {}".format(val)
        assert not torch.sum(torch.isinf(val)), "Inf occured in Coupling logdet {}".format(val)
        return val  # .unsqueeze(1)

    def forward(self, x, c):
        # first of all split data
        x1 = x[:, :self.dima, :, :]
        x2 = x[:, self.dima:, :, :]

        self.vals1 = self.s1(torch.cat([x2, c], dim=1))
        v1 = x1 * self.vals1.exp() + self.t1(torch.cat([x2, c], dim=1))
        self.vals2 = self.s2(torch.cat([v1, c], dim=1))
        v2 = x2 * self.vals2.exp() + self.t2(torch.cat([v1, c], dim=1))

        v = torch.cat([v1, v2], dim=1)

        return v

    def reverse(self, x, c):
        # first of all split data
        x1 = x[:, :self.dima, :, :]
        x2 = x[:, self.dima:, :, :]

        v2 = (x2 - self.t2(torch.cat([x1, c], dim=1))) * self.s2(torch.cat([x1, c], dim=1)).neg().exp()
        v1 = (x1 - self.t1(torch.cat([v2, c], dim=1))) * self.s1(torch.cat([v2, c], dim=1)).neg().exp()

        v = torch.cat([v1, v2], dim=1)

        return v


# copied from https://github.com/CompVis/net2net/blob/master/net2net/modules/flow/blocks.py
class InvertibleShuffleLayer(nn.Module):
    def __init__(self, nf):
        super(InvertibleShuffleLayer, self).__init__()

        idx = torch.randperm(nf)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def logdet(self):
        return 0.0

    def forward(self, x):
        return x[:, self.forward_shuffle_idx, ...]

    def reverse(self, x):
        return x[:, self.backward_shuffle_idx, ...]


class InvertibleConv1d(nn.Module):
    def __init__(self, nf):
        super(InvertibleConv1d, self).__init__()

        weight = torch.randn(nf, nf)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def logdet(self):
        return torch.slogdet(self.weight.squeeze().double())[1].float()

    def forward(self, x):
        return F.conv2d(x, self.weight)

    def reverse(self, x):
        return F.conv2d(x, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvertibleConvLU1d(nn.Module):
    def __init__(self, nf):
        super(InvertibleConvLU1d, self).__init__()

        self.nf = nf

        # random rotation matrix
        w_init = np.linalg.qr(np.random.randn(nf, nf))[0].astype(np.float32)
        p, l, u = alg.lu(w_init)
        s = np.diag(u)
        u = np.triu(u, k=1)

        lmask = np.tril(np.ones_like(w_init), -1)

        self.register_buffer("permutated", torch.FloatTensor(p))
        self.register_buffer("sign_s", torch.FloatTensor(np.sign(s)))
        self.register_buffer("lmask", torch.FloatTensor(lmask))
        self.register_buffer("umask", torch.FloatTensor(lmask.T))
        self.register_buffer("eye", torch.FloatTensor(np.eye(nf)))

        self.l = nn.Parameter(torch.FloatTensor(l))
        self.u = nn.Parameter(torch.FloatTensor(u))
        self.log_s = nn.Parameter(torch.FloatTensor(np.log(np.abs(s))))

    def logdet(self):
        val = torch.sum(self.log_s) * torch.ones(self.b, self.h, self.w).to(self.log_s.device)
        assert not torch.sum(torch.isnan(val)), "Nan occured in InvConv logdet {} {}".format(val, self.log_s)
        assert not torch.sum(torch.isinf(val)), "Inf occured in InvConv logdet {} {}".format(val, self.log_s)
        # val = val.view(1,1,1,1)
        return val

    def forward(self, x):
        self.b = x.size(0)
        self.h = x.size(2)
        self.w = x.size(3)
        wl = self.l * self.lmask + self.eye
        wu = self.u * self.umask + torch.diag(self.sign_s * torch.exp(self.log_s))
        weight = torch.matmul(self.permutated, torch.matmul(wl, wu)).view(self.nf, self.nf, 1, 1)

        return F.conv2d(x, weight)

    def reverse(self, x):
        wl = self.l * self.lmask + self.eye
        wu = self.u * self.umask + torch.diag(self.sign_s * torch.exp(self.log_s))
        weight = torch.matmul(torch.inverse(wu), torch.matmul(torch.inverse(wl),
                                                              torch.inverse(self.permutated))).view(self.nf, self.nf, 1,
                                                                                                    1)

        return F.conv2d(x, weight)


class InvertibleBlock(nn.Module):
    def __init__(self, nf, nf_hidden, normtype="act", acttype="none", alpha=1.0, use_conv=False, k=3, p=1):
        super(InvertibleBlock, self).__init__()

        self.use_conv = use_conv
        self.cacl = AffineCouplingLayer2d(nf, nf_hidden, k=k, p=p)
        self.norm = InvertibleNormLayer2d(nf, type=normtype)
        self.act = InvertibleActivationLayer(type=acttype, alpha=alpha)
        self.shuffle = InvertibleShuffleLayer(nf)
        self.conv = InvertibleConvLU1d(nf)

    def logdet(self):
        # print("test before")
        val = self.norm.logdet() + self.act.logdet() + self.cacl.logdet()
        if self.use_conv:
            val += self.conv.logdet()
        # print("test after {}".format(val.size()))
        # print(self.norm.logdet(), self.cacl.logdet(), self.conv.logdet())
        assert not torch.sum(torch.isnan(val)), "Nan occured in Block logdet {}".format(val)
        assert not torch.sum(torch.isinf(val)), "Inf occured in Block logdet {}".format(val)
        return val

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.cacl(x)
        if self.use_conv:
            x = self.conv(x)
        else:
            x = self.shuffle(x)

        return x

    def reverse(self, x):
        if self.use_conv:
            x = self.conv.reverse(x)
        else:
            x = self.shuffle.reverse(x)
        x = self.cacl.reverse(x)
        x = self.act.reverse(x)
        x = self.norm.reverse(x)

        return x


class ConditionalInvertibleBlock(nn.Module):
    def __init__(self, nf, nf_c, nf_hidden, normtype="act", acttype="none", alpha=1.0,
                 use_conv=False, k=3, p=1):
        super(ConditionalInvertibleBlock, self).__init__()

        self.use_conv = use_conv
        self.cacl = ConditionalAffineCouplingLayer2d(nf, nf_c, nf_hidden, k=k, p=p)
        self.norm = InvertibleNormLayer2d(nf, type=normtype)
        self.act = InvertibleActivationLayer(type=acttype, alpha=alpha)
        self.shuffle = InvertibleShuffleLayer(nf)
        self.conv = InvertibleConvLU1d(nf)

    def logdet(self):
        # print("test before")
        val = self.norm.logdet() + self.act.logdet() + self.cacl.logdet()
        if self.use_conv:
            val += self.conv.logdet()
        # print("test after {}".format(val.size()))
        # print(self.norm.logdet(), self.cacl.logdet(), self.conv.logdet())
        assert not torch.sum(torch.isnan(val)), "Nan occured in Block logdet {}".format(val)
        assert not torch.sum(torch.isinf(val)), "Inf occured in Block logdet {}".format(val)
        return val

    def forward(self, x, c):
        x = self.norm(x)
        x = self.act(x)
        x = self.cacl(x, c)
        if self.use_conv:
            x = self.conv(x)
        else:
            x = self.shuffle(x)

        return x

    def reverse(self, x, c):
        if self.use_conv:
            x = self.conv.reverse(x)
        else:
            x = self.shuffle.reverse(x)
        x = self.cacl.reverse(x, c)
        x = self.act.reverse(x)
        x = self.norm.reverse(x)

        return x


class InvertibleNet(nn.Module):
    def __init__(self, nf, nf_hidden, depth=10, norm="act", act="lrelu", use_conv=False, k=3, p=1):
        super(InvertibleNet, self).__init__()

        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(InvertibleBlock(nf, nf_hidden, norm, act, use_conv=use_conv, k=k, p=p))

    def logdet(self):
        logdet = 0.0
        for i in range(self.depth):
            logdet += self.blocks[i].logdet()

        return logdet

    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)

        return x

    def reverse(self, x):
        for i in reversed(range(self.depth)):
            x = self.blocks[i].reverse(x)

        return x


class ConditionalInvertibleNet(nn.Module):
    def __init__(self, config):
        super(ConditionalInvertibleNet, self).__init__()
        self.config = config

        depth = self.config['depth']
        nf = self.config['flow_in_channels']
        nf_c = self.config['h_channels']
        nf_hidden = self.config['flow_mid_channels']
        norm = self.config['norm']
        act = self.config['act']
        use_conv =self.config['invertible1x1']


        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConditionalInvertibleBlock(nf, nf_c, nf_hidden, norm, act,
                                                          use_conv=use_conv, k=3, p=1))

    def logdet(self):
        logdet = 0.0
        for i in range(self.depth):
            logdet += self.blocks[i].logdet()

        return logdet

    def forward(self, x, c):
        for i in range(self.depth):
            x = self.blocks[i](x, c)

        return x

    def reverse(self, x, c):
        for i in reversed(range(self.depth)):
            x = self.blocks[i].reverse(x, c)

        return x