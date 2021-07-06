# all the modules taken from https://github.com/XuezheMax/macow/tree/master
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from models.modules.INN.modules import NICEConvBlock, NICESelfAttnBlock, Conv2dWeightNorm, ShiftedConv2d
#from models.modules.INN.flow_blocks import Shuffle

class Conv1x1Flow(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.weight =nn. Parameter(torch.Tensor(in_channels, in_channels))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    def forward(self, input, reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if not reverse:
            batch, channels, H, W = input.size()
            out = F.conv2d(input, self.weight.view(self.in_channels, self.in_channels, 1, 1))
            _, logdet = torch.slogdet(self.weight)
            return out, logdet.mul(H * W)
        else:
            #batch, channels, H, W = input.size()
            out = F.conv2d(input, self.weight_inv.view(self.in_channels, self.in_channels, 1, 1))
            # _, logdet = torch.slogdet(self.weight_inv)
            return out


class MaCowStep(nn.Module):
    """
    A step of Macow Flows with 4 Macow Unit and a Glow step
    """
    def __init__(self, in_channels, kernel_size, hidden_channels, s_channels,num_units=2, scale=True,
                 coupling_type='conv', slice=None, heads=1, pos_enc=True, dropout=0.0):
        super().__init__()
        #num_units = 2
        units = [MaCowUnit(in_channels, kernel_size, s_channels, scale=scale, hidden_channels=hidden_channels) for _ in range(num_units)]
        self.units = nn.ModuleList(units)
        self.glow_step = GlowStep(in_channels, hidden_channels=hidden_channels, s_channels=s_channels, scale=scale,
                                  coupling_type=coupling_type, slice=slice, heads=heads, pos_enc=pos_enc, dropout=dropout)

    def sync(self):
        self.glow_step.sync()

    def forward(self, input, s=None, reverse=False):
        if not reverse:
            logdet_accum = input.new_zeros(input.size(0))
            out = input
            for unit in self.units:
                out, logdet = unit(out, s=s)
                logdet_accum = logdet_accum + logdet
            out, logdet = self.glow_step(out, s=s)
            logdet_accum = logdet_accum + logdet
            return out, logdet_accum
        else:
            out = self.glow_step(input, s=s,reverse=True)
            for unit in reversed(self.units):
                out = unit(out, s=s,reverse=True)
                #logdet_accum = logdet_accum + logdet
            return out



class ActNorm2dFlow(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.log_scale = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        self.reset_parameters()

        self.register_buffer('initialized', torch.tensor(0,dtype=torch.uint8))

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    def forward(self, input, reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if not reverse:

            batch, channels, H, W = input.size()
            out = input * self.log_scale.exp() + self.bias
            logdet = self.log_scale.sum(dim=0).squeeze(1).mul(H * W)
            if self.initialized.item() == 0:
                self.init(input)
                self.initialized.fill_(1)
                out = input * self.log_scale.exp() + self.bias
                logdet = self.log_scale.sum(dim=0).squeeze(1).mul(H * W)
            return out, logdet
        else:
            batch, channels, H, W = input.size()
            out = input - self.bias
            out = out.div(self.log_scale.exp() + 1e-8)
            #logdet = self.log_scale.sum(dim=0).squeeze(1).mul(H * -W)
            return out#,logdet

    def init(self, data, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            batch, channels, H, W = data.size()
            out = data * self.log_scale.exp() + self.bias

            out = out.transpose(0, 1).contiguous().view(self.in_channels, -1)
            # [n_channels, 1, 1]
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)

class MCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, order):
        super().__init__()
        self.masked_conv = ShiftedConv2d(in_channels, hidden_channels, kernel_size, order=order, bias=True)
        self.conv1x1 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.ELU()

    def forward(self, x, s=None, shifted=True):
        c = self.masked_conv(x, shifted=shifted)
        if s is not None:
            c = c + s
        c = self.conv1x1(self.activation(c))
        return c


class MaskedConvFlow(nn.Module):
    """
    Masked Convolutional Flow
    """

    def __init__(self, in_channels, kernel_size, hidden_channels=None, s_channels=None, order='A', scale=True):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        # if hidden_channels is None:
        #     if in_channels <= 96:
        #         hidden_channels = 4 * in_channels
        #     else:
        #         hidden_channels = min(2 * in_channels, 512)
        out_channels = in_channels
        if scale:
            out_channels = out_channels * 2
        self.kernel_size = kernel_size
        self.order = order
        self.sigm = nn.Sigmoid()
        self.net = MCFBlock(in_channels, out_channels, kernel_size, hidden_channels, order)
        if s_channels is None or s_channels <= 0:
            self.s_conv = None
        else:
            self.s_conv = Conv2dWeightNorm(s_channels, hidden_channels, (3, 3), bias=True, padding=1)

    def calc_mu_and_scale(self, x: torch.Tensor, s=None, shifted=True):
        mu = self.net(x, s=s, shifted=shifted)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = self.sigm(log_scale + 2.)
        return mu, scale

    def forward(self, input, s=None, reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            s: Tensor
                conditional input (default: None)
        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if not reverse:
            if self.s_conv is not None:
                s = self.s_conv(s)
            mu, scale = self.calc_mu_and_scale(input, s=s)
            out = input
            if self.scale:
                out = out.mul(scale)
                logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
            else:
                logdet = mu.new_zeros(mu.size(0))
            out = out + mu
            return out, logdet
        else:
            if self.s_conv is not None:
                ss = self.s_conv(s)
            else:
                ss = s
            if self.order == 'A':
                out = self.backward_height(input, s=ss, reverse=False)
            elif self.order == 'B':
                out = self.backward_height(input, s=ss, reverse=True)
            elif self.order == 'C':
                # fixme this changes dimension of inputs which is not intended
                out = self.backward_width(input, s=ss, reverse=False)
            else:
                out = self.backward_width(input, s=ss, reverse=True)
            # _, logdet = self.forward(out, s=s)
            return out#, logdet.mul(-1.0)

    def backward_height(self, input: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cW = kW // 2
        out = input.new_zeros(batch, channels, H + kH, W + 2 * cW)

        itr = reversed(range(H)) if reverse else range(H)
        for h in itr:
            curr_h = h if reverse else h + kH
            s_h = h + 1 if reverse else h
            t_h = h + kH + 1 if reverse else h + kH
            # [batch, channels, kH, width+2*cW]
            out_curr = out[:, :, s_h:t_h]
            s_curr = None if s is None else s[:, :, h:h + 1]
            # [batch, channels, width]
            in_curr = input[:, :, h]

            # [batch, channels, 1, width]
            mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
            # [batch, channels, width]
            new_out = in_curr - mu.squeeze(2)
            if self.scale:
                new_out = new_out.div(scale.squeeze(2) + 1e-12)
            out[:, :, curr_h, cW:W + cW] = new_out

        out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
        return out

    def backward_width(self, input: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cH = kH // 2
        out = input.new_zeros(batch, channels, H + 2 * cH, W + kW)

        itr = reversed(range(W)) if reverse else range(W)
        for w in itr:
            curr_w = w if reverse else w + kW
            s_w = w + 1 if reverse else w
            t_w = w + kW + 1 if reverse else w + kW
            # [batch, channels, height+2*cH, kW]
            out_curr = out[:, :, :, s_w:t_w]
            s_curr = None if s is None else s[:, :, :, w:w + 1]
            # [batch, channels, height]
            in_curr = input[:, :, :, w]

            # [batch, channels, height, 1]
            mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
            # [batch, channels, height]
            new_out = in_curr - mu.squeeze(3)
            if self.scale:
                new_out = new_out.div(scale.squeeze(3) + 1e-12)
            out[:, :, cH:H + cH, curr_w] = new_out

        out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
        return out



class MaCowUnit(nn.Module):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """
    def __init__(self, in_channels, kernel_size, s_channels,hidden_channels, scale=True):
        super().__init__()
        self.actnorm1 = ActNorm2dFlow(in_channels)
        self.actnorm2 = ActNorm2dFlow(in_channels)
        self.conv1 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), s_channels=s_channels,hidden_channels=hidden_channels ,order='A', scale=scale)
        self.conv2 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), s_channels=s_channels,hidden_channels=hidden_channels, order='B', scale=scale)
        self.conv3 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), s_channels=s_channels,hidden_channels=hidden_channels, order='C', scale=scale)
        self.conv4 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), s_channels=s_channels,hidden_channels=hidden_channels, order='D', scale=scale)

    def forward(self, input, s=None,reverse=False):
        if not reverse:
            # ActNorm1
            out, logdet_accum = self.actnorm1(input)
            # MCF1
            out, logdet = self.conv1(out, s=s)
            logdet_accum = logdet_accum + logdet
            # MCF2
            out, logdet = self.conv2(out, s=s)
            logdet_accum = logdet_accum + logdet
            # ActNorm2
            out, logdet = self.actnorm2(out)
            logdet_accum = logdet_accum + logdet
            # MCF3
            out, logdet = self.conv3(out, s=s)
            logdet_accum = logdet_accum + logdet
            # MCF4
            out, logdet = self.conv4(out, s=s)
            logdet_accum = logdet_accum + logdet
            return out, logdet_accum
        else:
            # MCF4
            out = self.conv4(input, s=s,reverse=True)
            # MCF3
            out = self.conv3(out, s=s,reverse=True)
            # logdet_accum = logdet_accum + logdet
            # ActNorm2
            out = self.actnorm2(out,reverse=True)
            # logdet_accum = logdet_accum + logdet
            # MCF2
            out = self.conv2(out, s=s,reverse=True)
            # logdet_accum = logdet_accum + logdet
            # MCF1
            out = self.conv1(out, s=s,reverse=True)
            # logdet_accum = logdet_accum + logdet
            # ActNorm1
            out = self.actnorm1(out,reverse=True)
            # logdet_accum = logdet_accum + logdet
            return out


class GlowStep(nn.Module):
    """
    A step of Glow. A Conv1x1 followed with a NICE
    """
    def __init__(self, in_channels, hidden_channels, s_channels=0, scale=True,
                 coupling_type='conv', slice=None, heads=1, pos_enc=True, dropout=0.0):
        super().__init__()
        from models.modules.INN.flow_blocks import Shuffle
        self.actnorm = ActNorm2dFlow(in_channels)
        #self.conv1x1 = Conv1x1Flow(in_channels)
        self.conv1x1 = Shuffle(in_channels)
        self.coupling = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels,
                             scale=scale, type=coupling_type, slice=slice, heads=heads, pos_enc=pos_enc, dropout=dropout)


    def forward(self, input, s=None, reverse=False):
        if not reverse:
            out, logdet_accum = self.actnorm(input)

            out, logdet = self.conv1x1(out)
            logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling(out, s=s)
            logdet_accum = logdet_accum + logdet
            return out, logdet_accum
        else:
            out= self.coupling(input, s=s,reverse=True)

            out = self.conv1x1(out,reverse=True)
            # logdet_accum = logdet_accum + logdet

            out = self.actnorm(out,reverse=True)
            # logdet_accum = logdet_accum + logdet
            return out

class NICE(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, s_channels=None, scale=True, factor=2,
                 type='conv', slice=None, heads=1, pos_enc=True, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        if hidden_channels is None:
            hidden_channels = min(8 * in_channels, 512)
        out_channels = in_channels // factor
        in_channels = in_channels - out_channels
        self.z1_channels = in_channels
        if scale:
            out_channels = out_channels * 2
        if s_channels is None:
            s_channels = 0
        assert type in ['conv', 'self_attn']
        if type == 'conv':
            self.net = NICEConvBlock(in_channels, out_channels, hidden_channels, s_channels, dilation=1)
        else:
            assert slice is not None, 'slice should be given.'
            slice = _pair(slice)
            self.net = NICESelfAttnBlock(in_channels, out_channels, hidden_channels, s_channels,
                                         slice=slice, heads=heads, pos_enc=pos_enc, dropout=dropout)

        self.sigm = nn.Sigmoid()

    def calc_mu_and_scale(self, z1: torch.Tensor, s=None):
        mu = self.net(z1, s=s)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = self.sigm(log_scale + 2.)
        return mu, scale

    def init_net(self, z1: torch.Tensor, s=None, init_scale=1.0):
        mu = self.net.init(z1, s=s, init_scale=init_scale)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

    def forward(self, input, s=None,reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            s: Tensor
                conditional input (default: None)
        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, in_channels, H, W]
        if not reverse:
            z1 = input[:, :self.z1_channels]
            z2 = input[:, self.z1_channels:]
            mu, scale = self.calc_mu_and_scale(z1, s)
            if self.scale:
                z2 = z2.mul(scale)
                logdet = scale.log().view(z1.size(0), -1).sum(dim=1)
            else:
                logdet = z1.new_zeros(z1.size(0))
            z2 = z2 + mu
            return torch.cat([z1, z2], dim=1), logdet
        else:
            z1 = input[:, :self.z1_channels]
            z2 = input[:, self.z1_channels:]
            mu, scale = self.calc_mu_and_scale(z1, s)
            z2 = z2 - mu
            if self.scale:
                z2 = z2.div(scale + 1e-12)
            #     logdet = scale.log().view(z1.size(0), -1).sum(dim=1) * -1.0
            # else:
            #     logdet = z1.new_zeros(z1.size(0))

            return torch.cat([z1, z2], dim=1)



    def init(self, data: torch.Tensor, s=None, init_scale=1.0):
        # [batch, in_channels, H, W]
        z1 = data[:, :self.z1_channels]
        z2 = data[:, self.z1_channels:]
        mu, scale = self.init_net(z1, s=s, init_scale=init_scale)
        if self.scale:
            z2 = z2.mul(scale)
            logdet = scale.log().view(z1.size(0), -1).sum(dim=1)
        else:
            logdet = z1.new_zeros(z1.size(0))
        z2 = z2 + mu

        return torch.cat([z1, z2], dim=1), logdet


