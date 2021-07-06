import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.INN.modules import BasicFullyConnectedNet, ActNorm, \
    MixCDFParameterTemplate, MixLogCDF, MixLogPDF,inv_mixlogcdf, SpaceToDepth, DepthToSpace, BasicResNet
from models.modules.INN.macow import MaCowStep

class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option
        self.reshape = 'none'

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            mode = 'cond' if flow % 2 == 0 else 'normal'
            # mode = 'normal' if flow % 2 == 0 else 'normal'
            # mode = 'cond' if flow % 2 == 0 else 'cond'
            #mode = 'normal' if flow % self.n_flows//2 == 0 else 'cond'
            self.sub_layers.append(ConditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.cond_channels, self.mid_channels,
                                   self.num_blocks, activation=activation, mode=mode))
            if self.conditioning_option.lower() != "none":
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, embedding, reverse=False):
        hconds = list()
        hcond = embedding
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding)
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            hconds.append(hcond)
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x.unsqueeze(-1).unsqueeze(-1), logdet
        else:
            for i in reversed(range(self.n_flows)):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class ConditionalConvFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option
        self.reshape = 'none'

        self.sub_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(ConditionalConvDoubleCouplingFlowBlock(
                                   self.in_channels, self.cond_channels, self.mid_channels,
                                   self.num_blocks, activation=activation))

    def forward(self, x, embedding, reverse=False):
        hcond = embedding
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x, logdet_ = self.sub_layers[i](x,hcond)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x = self.sub_layers[i](x, hcond, reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class ConditionalDoubleVectorCouplingBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels//2+cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels//2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels//2+cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels//2) for _ in range(2)])

    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class ConditionalDoubleConvCouplingBlock(nn.Module):

    def __init__(self,in_channels, cond_channels, hidden_dim, depth=2):
        super().__init__()
        self.s = nn.ModuleList([
            BasicResNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels // 2,data_init=True,last_zero=True) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicResNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels // 2,data_init=True,last_zero=True) for _ in range(2)])


    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(scale.size(0),-1), dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x

class ConditionalDoubleVectorCouplingBlock_cond(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        super(ConditionalDoubleVectorCouplingBlock_cond, self).__init__()
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels//2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels//2) for _ in range(2)])

    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = xc #torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = xc #torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu", mode='normal'):
        super().__init__()
        __possible_activations = {"lrelu": InvLeakyRelu, "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        if mode == 'cond':
            self.coupling = ConditionalDoubleVectorCouplingBlock_cond(in_channels, cond_channels, hidden_dim, hidden_depth)
        else:
            self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels, cond_channels, hidden_dim, hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class ConditionalConvDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="none"):
        super().__init__()
        __possible_activations = {"lrelu": InvLeakyRelu, "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleConvCouplingBlock(in_channels,cond_channels,hidden_dim,depth=hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]



class OrthogonalPermute(nn.Module):
    """For some orthogonal matrix O: O^(-1) = O^T"""
    def __init__(self, in_channels, **kwargs):
        super(OrthogonalPermute, self).__init__()
        print('WARNING: OrthogonalPermute induces invertibility issues!?')
        self.in_channels = in_channels
        omatrix = torch.empty(in_channels, in_channels)
        nn.init.orthogonal_(omatrix)
        self.register_buffer('forward_orthogonal', omatrix)
        self.register_buffer('backward_orthogonal', omatrix.t())

    def forward(self, x, reverse=False):
        twodim = False
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
            twodim = True
        if not reverse:
            if not twodim:
                return F.conv2d(x, self.forward_orthogonal.unsqueeze(2).unsqueeze(3)), 0
            return F.conv2d(x, self.forward_orthogonal.unsqueeze(2).unsqueeze(3)).squeeze(), 0
        else:
            if not twodim:
                return F.conv2d(x, self.backward_orthogonal.unsqueeze(2).unsqueeze(3))
            return F.conv2d(x, self.backward_orthogonal.unsqueeze(2).unsqueeze(3)).squeeze()


class IgnoreLeakyRelu(nn.Module):
    """performs identity op."""
    def __init__(self, alpha=0.9):
        super().__init__()

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        h = input
        return h, 0.0

    def reverse(self, input):
        h = input
        return h


class InvLeakyRelu(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input*scaling
        return h, 0.0

    def reverse(self, input):
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input/scaling
        return h


class InvParametricRelu(InvLeakyRelu):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)


class UnconditionalFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, activation=activation)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalFlow2(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, data_init=False):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock2(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, data_init)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)

class UnconditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, activation="lrelu"):
        super().__init__()
        __possible_activations = {"lrelu": lambda: InvLeakyRelu(alpha=0.95), "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = DoubleVectorCouplingBlock(in_channels,
                                                  hidden_dim,
                                                  hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalFlatDoubleCouplingFlowBlock2(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, data_init=False):
        super().__init__()
        self.norm_layer = ActNorm(in_channels, logdet=True)
        coupling = DoubleVectorCouplingBlock3 if data_init else DoubleVectorCouplingBlock2
        self.coupling = coupling(in_channels,
                                 hidden_dim,
                                 hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)

class DoubleVectorCouplingBlock(nn.Module):
    """In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super(DoubleVectorCouplingBlock, self).__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
                                                       use_tanh=True) for _ in range(2)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
                                                       use_tanh=False) for _ in range(2)])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]



class DoubleVectorCouplingBlock2(nn.Module):
    """Support uneven inputs"""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
        ])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]



class DoubleVectorCouplingBlock3(nn.Module):
    """Support uneven inputs"""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, data_init=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, data_init=True),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, data_init=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, data_init=True),
        ])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class Reshape(nn.Module):

    def __init__(self,blocksize=2):
        super().__init__()
        self.blocksize=blocksize
        self.down = SpaceToDepth(self.blocksize)
        self.up = DepthToSpace(self.blocksize)

    def forward(self,x,reverse=False):
        if reverse:
            return self.up(x), 0.
        return self.down(x), 0.

class FLowSigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()
        self.soft = nn.Softplus()
    def forward(self,x, reverse=False):
        if reverse:
            y = -torch.log(torch.reciprocal(x) - 1.)
            logdet = - torch.log(x) - torch.log(1.-x)
            return y, torch.sum(logdet.reshape(logdet.size(0),-1), dim=1)
        else:
            y = self.sigm(x)
            logdet = - self.soft(x) - self.soft(-x)
            logdet = torch.sum(logdet.reshape(logdet.size(0),-1), dim=1)
            return y, logdet

class Inverse(nn.Module):

    def __init__(self,fn:nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self,x,reverse=False):
        return self.fn(x,not reverse)

class Invertible1x1Conv(nn.Module):
    ## taken from https://github.com/corenel/pytorch-glow/blob/master/network/module.py
    def __init__(self, dimension, lu_decomposition=False):
        """
        Invertible 1x1 convolulution layer
        :param num_channels: number of channels
        :type num_channels: int
        :param lu_decomposition: whether to use LU decomposition
        :type lu_decomposition: bool
        """
        super().__init__()
        self.logdet_factor = dimension[1]*dimension[2]
        num_channels = dimension[0]
        self.lu_decomposition = lu_decomposition
        if self.lu_decomposition:
            raise NotImplementedError()
        else:
            w_shape = [num_channels, num_channels]
            # Sample a random orthogonal matrix
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
            self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))

    def forward(self, x, reverse=False):
        """
        :param x: input
        :type x: torch.Tensor
        :param logdet: log determinant
        :type logdet:
        :param reverse: whether to reverse bias
        :type reverse: bool
        :return: output and logdet
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """

        if not reverse:
            dlogdet = torch.log(torch.abs(torch.det(self.weight))) * self.logdet_factor
            weight = self.weight.view(*self.weight.shape, 1, 1)
            z = F.conv2d(x, weight)
            return z, dlogdet
        else:
            weight = self.weight.inverse().view(*self.weight.shape, 1, 1)
            z = F.conv2d(x, weight)
            return z


# conv flows
class TupleFlip(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x,reverse=False):
        a,b = torch.chunk(x,2,1)
        if reverse:
            return torch.cat([b,a],dim=1)
        return torch.cat([b, a], dim=1), 0

class UnconditionalMixCDFConvFlow(nn.Module):

    def __init__(self,config):
        super().__init__()
        dimension = config["latent_dim"]
        hidden_dim = config["flow_mid_channels"]
        num_blocks = config["flow_hidden_depth"]
        heads = config["flow_attn_heads"]
        components = config["flow_cdf_components"]
        p_dropout = config["flow_p_drop"]
        self.n_flows = config["n_flows"]
        self.sub_layers = nn.ModuleList()
        self.reshape = config["reshape"]
        shuffle = config["shuffle"] if "shuffle" in config else True
        activation = config["activation"] if "activation" in config else "lrelu"

        if not shuffle:
            print("using invertible 1x1 conv in flow block!")
        self.preprocess = config["preproc"]
        if self.preprocess:
            self.preproc_fn = FLowSigmoid()

        reshape_id = int(self.n_flows // 2)
        if self.reshape != "none":
            self.n_flows += 1
            assert self.reshape in ["up","down"]

        if self.reshape =="up":
            reshape_fn = Inverse(Reshape(2))
            self.reshape_factor = 0.5
        elif self.reshape =="down":
            reshape_fn = Reshape(2)
            self.reshape_factor=2

        for i,flow in enumerate(range(self.n_flows)):
            if i == reshape_id and self.reshape != "none":
                self.sub_layers.append(reshape_fn)
                dimension = [int((self.reshape_factor**2)*dimension[0]),int(dimension[1]/self.reshape_factor),int(dimension[2]/self.reshape_factor)]
                hidden_dim = int((self.reshape_factor**2) * hidden_dim)
            else:
                self.sub_layers.append(UnconditionalMixCDFCouplingFlowBlock(
                    dimension,hidden_dim,
                    num_blocks, heads, components,p_dropout,
                    w_init=config["weight_init"] if "weight_init" in config else "data",
                shuffle=shuffle,activation=activation)
                )


    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            if self.preprocess:
                x, logdet_ = self.preproc_fn(x)
                logdet = logdet + logdet_
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
                if isinstance(x,tuple):
                    x = x[0]
            if self.preprocess:
                x, _ = self.preproc_fn(x,reverse=True)

            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalMaCowFLowBlock(nn.Module):

    def __init__(self, channels,kernel_size,hidden_channels,
                 scale,heads,coupling_type="conv",num_blocks=2, activation="lrelu"):
        super().__init__()

        __possible_activations = {"lrelu": InvLeakyRelu, "none": IgnoreLeakyRelu, "sigmoid":FLowSigmoid}

        self.norm_layer = ActNorm(channels,logdet=True)
        #dimension_coupling = (int(channels//2),H,W)
        self.coupling = MaCowStep(channels,kernel_size=kernel_size,
                                             hidden_channels=hidden_channels,s_channels=None,
                                             scale=scale,heads=heads,coupling_type=coupling_type,
                                             num_units=num_blocks)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(channels) #if shuffle else Invertible1x1Conv(dimension)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)

class UnconditionalMixCDFCouplingFlowBlock(nn.Module):

    def __init__(self, dimension, hidden_dim, num_blocks,heads, components,p_dropout=0., activation="lrelu",w_init="data",shuffle=True):
        super().__init__()

        __possible_activations = {"lrelu": InvLeakyRelu, "none": IgnoreLeakyRelu, "sigmoid":FLowSigmoid}
        in_channels, H, W = dimension
        self.norm_layer = ActNorm(in_channels,logdet=True)
        dimension_coupling = (int(in_channels//2),H,W)
        self.coupling = UnconditionalMixCDFCouplingBlock(dimension=dimension_coupling,hidden_dim=hidden_dim,
                                                         blocks=num_blocks,heads=heads,components=components,
                                                         p_dropout=p_dropout,w_init=w_init)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels) if shuffle else Invertible1x1Conv(dimension)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.activation(h, reverse=True)[0] if isinstance(self.activation,FLowSigmoid) \
                else self.activation(h,reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalMixCDFCouplingBlock(nn.Module):

    def __init__(self, dimension, hidden_dim, blocks, heads,components,p_dropout=0.,w_init="data"):
        super().__init__()
        self.dimension = dimension
        self.scale = Inverse(FLowSigmoid())
        self.affine_cdf_params = MixCDFParameterTemplate(dimension,hidden_dim,blocks,
                                                         heads,components,p_dropout=p_dropout,winit=w_init)

        self.mixlogcdf = MixLogCDF()
        self.mixlogpdf = MixLogPDF()

    def forward(self,x,h=None,reverse=False):
        if not reverse:
            x_ = torch.chunk(x, 2, dim=1)
            # split
            x_1,x_2 = x_[0],x_[1]
            # get transformation parameters
            s,t,ml_logits, ml_means, ml_logscales = self.affine_cdf_params(x_1)
            #transform x_2 using mxiture of logistic cumulative distribution functions
            y_2 = torch.exp(self.mixlogcdf(x_2,ml_logits, ml_means, ml_logscales))
            # inverse sigmoid as in the paper
            y_2, scale_logdet = self.scale(y_2)
            # affine transformation
            y_2 = y_2 * torch.exp(s) + t
            # logdet of mixlogcdf
            logdet = self.mixlogpdf(x_2,ml_logits, ml_means, ml_logscales,exp=False)
            # add logdet of affine transform
            logdet = logdet + s
            logdet = torch.sum(logdet.reshape(logdet.size(0),-1),dim=1) + scale_logdet
            y = torch.cat([x_1,y_2],dim=1)
            return y, logdet
        else:
            x_= torch.chunk(x,2,dim=1)
            x_1,x_2 = x_[0],x_[1]
            # transform params
            s, t, ml_logits, ml_means, ml_logscales = self.affine_cdf_params(x_1)
            y_2 = s.neg().exp() * (x_2 - t)
            y_2, _ = self.scale(y_2,reverse=True)
            y_2 = torch.clamp(y_2,min=0.00001,max=0.9999)
            y_2 = inv_mixlogcdf(y_2,ml_logits, ml_means, ml_logscales)

            out = torch.cat([x_1,y_2],dim=1)
            return out

def orth_correction(R):
    R[0] /= torch.norm(R[0])
    for i in range(1, R.shape[0]):

        R[i] -= torch.sum( R[:i].t() * torch.matmul(R[:i], R[i]), dim=1)
        R[i] /= torch.norm(R[i])

def correct_weights(module, grad_in, grad_out):

    module.back_counter += 1

    if module.back_counter > module.correction_interval:
        module.back_counter = np.random.randint(0, module.correction_interval) // 4
        orth_correction(module.weights.data)

class OrthogonalTransform(nn.Module):
    '''  '''

    def __init__(self, dims_in, correction_interval=256, clamp=5.):
        super().__init__()
        self.width = dims_in[0][0]
        self.clamp = clamp

        self.correction_interval = correction_interval
        self.back_counter = np.random.randint(0, correction_interval) // 2

        self.weights = torch.randn(self.width, self.width)
        self.weights = self.weights + self.weights.t()
        self.weights, S, V = torch.svd(self.weights)

        self.weights = nn.Parameter(self.weights)

        self.bias = nn.Parameter(0.05 * torch.randn(self.width))
        self.scaling = nn.Parameter(0.02 * torch.randn(self.width))

        self.register_backward_hook(correct_weights)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s/self.clamp))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s/self.clamp)

    def forward(self, x, reverse=False):
        if reverse:
            return [(x[0] / self.e(self.scaling) - self.bias).mm(self.weights.t())]
        logdet = torch.sum(self.log_e(self.scaling)).view(1,).expand(x[0].shape[0])
        return [(x[0].mm(self.weights) + self.bias) * self.e(self.scaling)], logdet


    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

class UnconditionalMaCow(nn.Module):

    def __init__(self,config):
        super().__init__()
        super().__init__()
        channels = config["flow_in_channels"]
        hidden_dim = config["flow_mid_channels"]
        num_blocks = config["flow_hidden_depth"]
        coupling_type = config["coupling_type"]
        kernel_size = config["kernel_size"]
        heads = config["flow_attn_heads"]
        scale= config["scale"]
        self.n_flows = config["n_flows"]
        self.reshape = "none"
        self.sub_layers = nn.ModuleList()

        for i, flow in enumerate(range(self.n_flows)):

            self.sub_layers.append(MaCowStep(channels,kernel_size=kernel_size,
                                             hidden_channels=hidden_dim,s_channels=None,
                                             scale=scale,heads=heads,coupling_type=coupling_type,
                                             num_units=num_blocks))

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
                if isinstance(x, tuple):
                    x = x[0]
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalMaCowFlow(nn.Module):

    def __init__(self,config):
        super().__init__()
        super().__init__()
        channels = config["flow_in_channels"]
        hidden_dim = config["flow_mid_channels"]
        num_blocks = config["flow_hidden_depth"]
        coupling_type = config["coupling_type"]
        kernel_size = config["kernel_size"]
        heads = config["flow_attn_heads"]
        scale= config["scale"]
        self.n_flows = config["n_flows"]
        self.reshape = "none"
        self.sub_layers = nn.ModuleList()

        for i, flow in enumerate(range(self.n_flows)):

            self.sub_layers.append(UnconditionalMaCowFLowBlock(channels=channels,kernel_size=kernel_size,hidden_channels=hidden_dim,
                                                               scale=scale,heads=heads,coupling_type=coupling_type,
                                                               num_blocks=num_blocks))

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
                if isinstance(x, tuple):
                    x = x[0]
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalExtendedLeapFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, delta_t=1.):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):

            self.sub_layers.append(UnconditionalExtendedLeapFrogBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, delta_t=delta_t)
                                   )

    def forward(self, x, v, reverse=False):
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, v, logdet_ = self.sub_layers[i](x,v)
                logdet = logdet + logdet_
            return x, v[:,:,None,None], logdet
        else:
            for i in reversed(range(self.n_flows)):
                x,v = self.sub_layers[i](x,v, reverse=True)
            return x,v[:,:,None,None]

    def reverse(self, out):
        return self(out, reverse=True)

class UnconditionalExtendedLeapFrogBlock(nn.Module):

    def __init__(self, in_channels, hidden_dim, hidden_depth,delta_t=1.):
        super().__init__()
        self.norm_layer_x = ActNorm(in_channels, logdet=True)
        self.norm_layer_v = ActNorm(in_channels, logdet=True)
        self.coupling = ExtendedLeapFrogCouplingBlock(in_channels,
                                                      hidden_dim,
                                                      hidden_depth,
                                                      delta_t)
        self.shuffle_x = Shuffle(in_channels)
        self.shuffle_v = Shuffle(in_channels)


    def forward(self, x, v, reverse=False):
        if not reverse:
            h = x
            hv = v
            logdet = 0.0
            h, ld = self.norm_layer_x(h)
            logdet += ld
            hv, ld = self.norm_layer_v(hv)
            logdet += ld
            h, hv, ld = self.coupling(h,hv)
            logdet += ld
            h, ld = self.shuffle_x(h)
            logdet += ld
            hv, ld = self.shuffle_v(hv)
            logdet += ld
            return h, hv, logdet
        else:
            h = x
            hv = v
            h = self.shuffle_x(h, reverse=True)
            hv = self.shuffle_v(hv, reverse=True)
            h,hv = self.coupling(h,hv, reverse=True)
            h = self.norm_layer_x(h, reverse=True)
            hv = self.norm_layer_v(hv, reverse=True)
            return h,hv

class UnconditionalLeapFlow(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows,delta_t=1.):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):

            self.sub_layers.append(UnconditionalLeapFrogBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks,delta_t=delta_t)
                                   )

    def forward(self, x, v, reverse=False):
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, v, logdet_ = self.sub_layers[i](x,v)
                logdet = logdet + logdet_
            return x, v[:,:,None,None], logdet
        else:
            for i in reversed(range(self.n_flows)):
                x,v = self.sub_layers[i](x,v, reverse=True)
            return x,v[:,:,None,None]

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalLeapFrogBlock(nn.Module):

    def __init__(self, in_channels, hidden_dim, hidden_depth,delta_t=1.):
        super().__init__()
        self.norm_layer_x = ActNorm(in_channels, logdet=True)
        self.norm_layer_v = ActNorm(in_channels, logdet=True)
        self.coupling = LeapFrogCouplingBlock(in_channels,
                                              hidden_dim,
                                              hidden_depth,
                                              delta_t=delta_t)
        self.shuffle_x = Shuffle(in_channels)
        self.shuffle_v = Shuffle(in_channels)


    def forward(self, x, v, reverse=False):
        if not reverse:
            h = x
            hv = v
            logdet = 0.0
            h, ld = self.norm_layer_x(h)
            logdet += ld
            hv, ld = self.norm_layer_v(hv)
            logdet += ld
            h, hv, ld = self.coupling(h,hv)
            logdet += ld
            h, ld = self.shuffle_x(h)
            logdet += ld
            hv, ld = self.shuffle_v(hv)
            logdet += ld
            return h, hv, logdet
        else:
            h = x
            hv = v
            h = self.shuffle_x(h, reverse=True)
            hv = self.shuffle_v(hv, reverse=True)
            h,hv = self.coupling(h,hv, reverse=True)
            h = self.norm_layer_x(h, reverse=True)
            hv = self.norm_layer_v(hv, reverse=True)
            return h,hv



class LeapFrogCouplingBlock(nn.Module):

    def __init__(self, in_channels,hidden_dim,hidden_depth=2,delta_t=1.):
        super().__init__()
        self.delta_t = delta_t
        self.grad_u = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, data_init=True),
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, data_init=True, last_zero=True),
        ])
        self.scale_p = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, data_init=True),
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, data_init=True, last_zero=True),
        ])

        self.actnorm_v = ActNorm(num_features=in_channels,logdet=True)
        self.actnorm_x = ActNorm(num_features=in_channels,logdet=True)


    def forward(self, x, v,reverse=False):
        assert len(x.shape) == 4
        assert len(v.shape) == 2
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:

            # update v_n --> v'

            # coupling update on x
            logdet = 0
            for i in range(len(self.grad_u)):
                # update v_n --> v'
                v_prime = v - .5* self.delta_t * self.grad_u[i](x)
                # update x_n --> x_(n+1)
                x = x + self.scale_p[i](v_prime) * self.delta_t
                # update v' --> v_(n+1)
                v = v_prime - .5* self.delta_t * self.grad_u[i](x)
                if i == 0:
                    v, logdet_ = self.actnorm_v(v)
                    logdet = logdet+logdet_
                    x, logdet_ = self.actnorm_x(x)
                    logdet = logdet + logdet_

            logdet = torch.zeros_like(v).sum(-1)

            return x[:,:,None,None], v, logdet
        else:


            for i in reversed(range(len(self.grad_u))):
                if i==0:
                    v = self.actnorm_v(v,reverse=True)
                    x = self.actnorm_x(x,reverse=True)
                v_prime = v +.5* self.delta_t * self.grad_u[i](x)
                # update x_n --> x_(n+1)
                x_ = x - self.scale_p[i](v_prime)* self.delta_t
                # update v' --> v_(n+1)
                v = v_prime + .5* self.delta_t * self.grad_u[i](x)



            return x[:,:,None,None], v


class ExtendedLeapFrogCouplingBlock(nn.Module):
    """Support uneven inputs"""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2,delta_t=1.):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.delta_t = delta_t
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True,last_zero=True),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False,data_init=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False,data_init=True,last_zero=True),
        ])


        self.q = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True,last_zero=True),
        ])

        self.f = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False,data_init=True),
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False,data_init=True,last_zero=True),
        ])

        self.v = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True),
            BasicFullyConnectedNet(dim=in_channels, out_dim=in_channels, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True,data_init=True,last_zero=True),
        ])




    def forward(self, x, v,reverse=False):
        assert len(x.shape) == 4
        assert len(v.shape) == 2
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:

            # update v_n --> v'
            scale_v1 = self.v[0](x)
            v_prime = v * (.5 * scale_v1).exp() - .5 * self.delta_t * self.f[0](x)
            logdet = .5 * torch.sum(scale_v1.view(x.size(0), -1), dim=1)
            # coupling update on x
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                    v_prime = torch.cat(torch.chunk(v_prime, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                v_prime = torch.chunk(v_prime, 2, dim=1)
                scale_x = self.s[i](x[idx_apply])
                scale_v = self.q[i](x[idx_apply])
                x_ = x[idx_keep] * (scale_x.exp()) + self.t[i](x[idx_apply]) + (scale_v.exp()) * self.delta_t * v_prime[idx_keep]
                x = torch.cat((x[idx_apply], x_), dim=1)
                v_prime = torch.cat([v_prime[idx_apply],v_prime[idx_keep]], dim=1)
                logdet_ = torch.sum(scale_x.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_

            # update v' --> v_(n+1)
            scale_v2 = self.v[1](x)
            v = v_prime * (.5 * scale_v2).exp() - .5 * self.delta_t* self.f[1](x)
            logdet = logdet + .5 * torch.sum(scale_v2.view(x.size(0), -1), dim=1)

            return x[:,:,None,None], v, logdet
        else:

            v_prime = (v +.5* self.delta_t*self.f[1](x)) * ((.5 * self.v[1](x)).neg().exp())

            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                    v_prime = torch.cat(torch.chunk(v_prime, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                v_prime = torch.chunk(v_prime, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply]) - self.q[i](x[idx_apply])* self.delta_t * v_prime[idx_keep]) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
                v_prime = torch.cat([v_prime[idx_apply],v_prime[idx_keep]], dim=1)


            v = (v_prime +.5* self.delta_t*self.f[0](x)) * ((.5 * self.v[0](x)).neg().exp())

            return x[:,:,None,None], v


class HMCBlock(nn.Module):

    def __init__(self, channels_p, channels_q, delta_t, hamiltonian_mode='stack'):
        super().__init__()
        assert hamiltonian_mode in ['stack','sum']
        # dim1 = (in_channels // 2) + (in_channels % 2)
        # dim2 = in_channels // 2
        self.delta_t = delta_t
        self.hamiltonian_mode = hamiltonian_mode

        if self.hamiltonian_mode == 'stack':
            # single hamiltonian net
            self.h_net = BasicFullyConnectedNet()
        else:
            self.h_p = BasicFullyConnectedNet()


