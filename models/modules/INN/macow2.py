import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy.linalg as alg



from models.modules.INN.flow_blocks import Shuffle, InvLeakyRelu, ActNorm, UnconditionalMixCDFCouplingBlock,Reshape, Inverse
from models.modules.INN.macow_utils import Additive,Affine,ReLU,NLSQ,SymmELU, NICEConvBlock,MCFBlock
from models.modules.INN.modules import GatedConv2d
from models.modules.autoencoders.util import Conv2dBlock,Conv2dTransposeBlock


def split2d(x: torch.Tensor, z1_channels):
    z1 = x[:, :z1_channels]
    z2 = x[:, z1_channels:]
    return z1, z2


def unsplit2d(xs):
    # [batch, channels, heigh, weight]
    return torch.cat(xs, dim=1)

class MaskedConvFlow(nn.Module):
    """
    Masked Convolutional Flow
    """

    def __init__(self, in_channels, kernel_size, hidden_channels=None, h_channels=None,
                 h_type=None, activation='relu', order='A', transform='affine', alpha=1.0,
                 p_dropout=0.):
        super().__init__()
        self.in_channels = in_channels
        self.cond = h_channels is not None and h_channels > 0
        if hidden_channels is None:
            if in_channels <= 96:
                hidden_channels = 4 * in_channels
            else:
                hidden_channels = min(2 * in_channels, 512)
        out_channels = in_channels
        assert transform in ['additive', 'affine', 'relu', 'nlsq', 'symm_elu']
        if transform == 'additive':
            self.transform = Additive()
            self.analytic_bwd = True
        elif transform == 'affine':
            self.transform = Affine(dim=1, alpha=alpha)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'relu':
            self.transform = ReLU(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'nlsq':
            self.transform = NLSQ(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 5
        elif transform == 'symm_elu':
            self.transform = SymmELU(dim=1)
            self.analytic_bwd = False
            out_channels = out_channels * 2
        else:
            raise ValueError('unknown transform: {}'.format(transform))
        self.kernel_size = kernel_size
        self.order = order
        self.net = MCFBlock(in_channels, out_channels, kernel_size,
                            hidden_channels, order, activation,
                            h_channels=h_channels if self.cond else None,
                            p_dropout=p_dropout)




        # assert h_type in [None, 'local_linear', 'global_linear', 'global_attn']
        # if h_type is None:
        #     assert h_channels is None or h_channels == 0
        #     self.h_net = None
        # elif h_type == 'local_linear':
        #     self.h_net = LocalLinearCondNet(h_channels, hidden_channels, kernel_size=3)
        # elif h_type == 'global_linear':
        #     # TODO remove global linear
        #     self.h_net = GlobalLinearCondNet(h_channels, hidden_channels)
        # elif h_type == 'global_attn':
        #     # TODO add global attn
        #     self.h_net = None
        # else:
        #     raise ValueError('unknown conditional transform: {}'.format(h_type))

    def calc_params(self, x: torch.Tensor, h=None, shifted=True):
        params = self.net(x, h=h, shifted=shifted)
        return params

    # def init_net(self, x, h=None, init_scale=1.0):
    #     params = self.net.init(x, h=h, init_scale=init_scale)
    #     return params

    def forward(self, input: torch.Tensor, h=None,reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # strategy is to compute the scaling and shift
        if self.cond: assert h is not None
        # if self.cond:
        #     assert h is not None
        #     input_params = h
        #     h = None
        # else:
        #     input_params = input
        if not reverse:
            params = self.transform.calc_params(self.calc_params(input, h=h))
            out, logdet = self.transform.fwd(input, params)
            return out, logdet
        else:
            # fixme continue here, params of affine transform are calculated with conditioning, if such one is given
            # this has also tp be done for the backward pass. Look threrefore, whow the backward pass works and adapt accordingly
            if self.analytic_bwd:
                return self.backward_analytic(input, h=h)
            else:
                raise NotImplementedError("Not yet implemented for conditional input")
                #return self.backward_iterative(input, h=h)




    def backward_analytic(self, z: torch.Tensor, h=None):
        # if self.h_net is not None:
        #     bs, _, H, W = z.size()
        #     h = self.h_net(h)
        #     hh = h + h.new_zeros(bs, 1, H, W)
        # else:
        #     h = hh = None
        if self.order == 'A':
            out = self.backward_height(z, hh=h, reverse=False)
        elif self.order == 'B':
            out = self.backward_height(z, hh=h, reverse=True)
        elif self.order == 'C':
            out = self.backward_width(z, hh=h, reverse=False)
        else:
            out = self.backward_width(z, hh=h, reverse=True)

        # params = self.transform.calc_params(self.calc_params(input_params, h=h))
        # _, logdet = self.transform.fwd(out, params)
        return out#, logdet.mul(-1.0)

    def backward_iterative(self, z: torch.Tensor, h=None, maxIter=100):
        if self.h_net is not None:
            h = self.h_net(h)
        else:
            h = None

        z_org = z
        eps = 1e-6
        for iter in range(maxIter):
            params = self.transform.calc_params(self.calc_params(z, h=h))
            new_z, logdet = self.transform.bwd(z, params)
            new_z = z_org - new_z
            diff = torch.abs(new_z - z).max().item()
            z = new_z
            if diff < eps:
                break

        #params = self.transform.calc_params(self.calc_params(z, h=h))
        #z_recon, logdet = self.transform.fwd(z, params)
        return z#, logdet * -1.0

    def backward_height(self, input: torch.Tensor, hh=None, reverse=False) -> torch.Tensor:
        # batch, channels, H, W = input.size()
        #
        # kH, kW = self.kernel_size
        # cW = kW // 2
        # out = input.new_zeros(batch, channels, H + kH, W + 2 * cW)
        # # bs_p, c_p, H_p, W_p = params_input.size()
        # # assert bs_p == batch and H_p == H and W == W_p
        # # params_input = input.new_zeros(batch, c_p, H + kH, W + 2 * cW)
        #
        # itr = reversed(range(H)) if reverse else range(H)
        # for h in itr:
        #     curr_h = h if reverse else h + kH
        #     s_h = h + 1 if reverse else h
        #     t_h = h + kH + 1 if reverse else h + kH
        #     # [batch, channels, kH, width+2*cW]
        #     out_curr = out[:, :, s_h:t_h]
        #     #out_curr = params_input[:, :, s_h:t_h]
        #     hh_curr = None if hh is None else hh[:, :, h:h + 1]
        #     # [batch, channels, width]
        #     in_curr = input[:, :, h]
        #
        #     # [batch, channels, 1, width]
        #     params = self.calc_params(out_curr, h=hh_curr, shifted=False)
        #     params = self.transform.calc_params(params.squeeze(2))
        #     # [batch, channels, width]
        #     new_out, _ = self.transform.bwd(in_curr, params)
        #     out[:, :, curr_h, cW:W + cW] = new_out
        #
        # out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
        # return out
        # ref_out = params_input if self.cond else input
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
            hh_curr = None if hh is None else hh[:, :, h:h + 1]
            # [batch, channels, width]
            in_curr = input[:, :, h]

            # [batch, channels, 1, width]
            params = self.calc_params(out_curr, h=hh_curr, shifted=False)
            params = self.transform.calc_params(params.squeeze(2))
            # [batch, channels, width]
            new_out, _ = self.transform.bwd(in_curr, params)
            out[:, :, curr_h, cW:W + cW] = new_out

        out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
        return out

    def backward_width(self, input: torch.Tensor, hh=None, reverse=False) :
        # batch, channels, H, W = input.size()
        #
        # kH, kW = self.kernel_size
        # cH = kH // 2
        # out = input.new_zeros(batch, channels, H + 2 * cH, W + kW)
        # # bs_p, c_p, H_p, W_p = params_input.size()
        # # assert bs_p == batch and H_p == H and W == W_p
        # # out = input.new_zeros(batch, channels, H + 2 * cH, W + kW)
        #
        # itr = reversed(range(W)) if reverse else range(W)
        # for w in itr:
        #     curr_w = w if reverse else w + kW
        #     s_w = w + 1 if reverse else w
        #     t_w = w + kW + 1 if reverse else w + kW
        #     # [batch, channels, height+2*cH, kW]
        #     out_curr = params_input[:, :, :, s_w:t_w]
        #     hh_curr = None if hh is None else hh[:, :, :, w:w + 1]
        #     # [batch, channels, height]
        #     in_curr = input[:, :, :, w]
        #
        #     # [batch, channels, height, 1]
        #     params = self.calc_params(out_curr, h=hh_curr, shifted=False)
        #     params = self.transform.calc_params(params.squeeze(3))
        #     # [batch, channels, height]
        #     new_out, _ = self.transform.bwd(in_curr, params)
        #     out[:, :, cH:H + cH, curr_w] = new_out
        #
        # out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
        # return out
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
            hh_curr = None if hh is None else hh[:, :, :, w:w + 1]
            # [batch, channels, height]
            in_curr = input[:, :, :, w]

            # [batch, channels, height, 1]
            params = self.calc_params(out_curr, h=hh_curr, shifted=False)
            params = self.transform.calc_params(params.squeeze(3))
            # [batch, channels, height]
            new_out, _ = self.transform.bwd(in_curr, params)
            out[:, :, cH:H + cH, curr_w] = new_out

        out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
        return out


class NICE2d(nn.Module):
    """
    NICE Flow for 2D image data
    """
    def __init__(self, in_channels, hidden_channels=None, h_channels=0,
                 split_type='continuous', order='up', factor=2, transform='affine', alpha=1.0,
                 type='conv', h_type=None, activation='relu', normalize=None, num_groups=None,
                 attention=False,heads=4,spatial_size=8,cond_conv=False,cond_conv_hidden_channels=None,p_dropout=0.):
        super().__init__()

        self.in_channels = in_channels
        self.factor = factor
        assert split_type in ['continuous', 'skip']
        if split_type == 'skip':
            assert factor == 2
            if in_channels % factor == 1:
                split_type = 'continuous'
        assert order in ['up', 'down']
        self.split_type = split_type
        self.up = order == 'up'

        if hidden_channels is None:
            hidden_channels = min(8 * in_channels, 512)

        out_channels = in_channels // factor
        in_channels = in_channels - out_channels
        self.z1_channels = in_channels if self.up else out_channels

        assert transform in ['additive', 'affine', 'relu', 'nlsq', 'symm_elu']
        if transform == 'additive':
            self.transform = Additive()
            self.analytic_bwd = True
        elif transform == 'affine':
            self.transform = Affine(dim=1, alpha=alpha)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'relu':
            self.transform = ReLU(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'nlsq':
            self.transform = NLSQ(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 5
        elif transform == 'symm_elu':
            self.transform = SymmELU(dim=1)
            self.analytic_bwd = False
            out_channels = out_channels * 2
        else:
            raise ValueError('unknown transform: {}'.format(transform))


        assert type in ['conv']
        if type == 'conv':
            self.net = NICEConvBlock(in_channels, out_channels, hidden_channels, activation,
                                     normalize=normalize, num_groups=num_groups, h_channels=h_channels,
                                     attention=attention,heads=heads,spatial_size=spatial_size,
                                     cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                     p_dropout=p_dropout)

        # assert h_type in [None, 'local_linear', 'global_linear', 'global_attn']
        # if h_type is None:
        #     assert h_channels == 0
        #     self.h_net = None
        # elif h_type == 'local_linear':
        #     self.h_net = LocalLinearCondNet(h_channels, hidden_channels, kernel_size=3)
        # elif h_type == 'global_linear':
        #     self.h_net = GlobalLinearCondNet(h_channels, hidden_channels)
        # elif h_type == 'global_attn':
        #     self.h_net = GlobalAttnCondNet(h_channels, in_channels, hidden_channels)
        # else:
        #     raise ValueError('unknown conditional transform: {}'.format(h_type))

    def split(self, z):
        split_dim = 1
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim)
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            return z1, z2
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = 1
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor, h=None):
        params = self.net(z, h=h)
        return params


    def forward(self, input: torch.Tensor, h=None,reverse=False):
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if not reverse:
            # [batch, length, in_channels]
            z1, z2 = self.split(input)
            # [batch, length, features]
            z, zp = (z1, z2) if self.up else (z2, z1)

            # if self.h_net is not None:
            #     h = self.h_net(h, x=z)
            # else:
            #     h = None

            params = self.transform.calc_params(self.calc_params(z, h=h))
            zp, logdet = self.transform.fwd(zp, params)

            z1, z2 = (z, zp) if self.up else (zp, z)
            return self.unsplit(z1, z2), logdet
        else:
            if self.analytic_bwd:
                return self.backward_analytic(input, h=h)
            else:
                return self.backward_iterative(input, h=h)



    def backward_analytic(self, z: torch.Tensor, h=None):
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        # if self.h_net is not None:
        #     h = self.h_net(h, x=z)
        # else:
        #     h = None

        params = self.transform.calc_params(self.calc_params(z, h=h))
        zp, logdet = self.transform.bwd(zp, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2)#, logdet

    def backward_iterative(self, z: torch.Tensor, h=None, maxIter=100):
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        # if self.h_net is not None:
        #     h = self.h_net(h, x=z)
        # else:
        #     h = None

        params = self.transform.calc_params(self.calc_params(z, h=h))
        zp_org = zp
        eps = 1e-6
        for iter in range(maxIter):
            new_zp, logdet = self.transform.bwd(zp, params)
            new_zp = zp_org - new_zp
            diff = torch.abs(new_zp - zp).max().item()
            zp = new_zp
            if diff < eps:
                break

        _, logdet = self.transform.fwd(zp, params)
        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2)#, logdet * -1.0

class ActNorm2dFlow(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.log_scale = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)


    def forward(self, input: torch.Tensor, reverse=False):
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        if not reverse:
            if self.initialized.item() == 0:
                self.init(input)
                self.initialized.fill_(1)

            batch, channels, H, W = input.size()
            # [channels, 1, 1]
            log_scale = self.log_scale
            bias = self.bias
            out = input * log_scale.exp() + bias
            logdet = log_scale.view(1, channels).sum(dim=1).mul(H * W)
            logdet = logdet * torch.ones(input.shape[0]).type_as(input)
            return out, logdet
        else:
            batch, channels, H, W = input.size()
            # [channels, 1, 1]
            log_scale = self.log_scale
            bias = self.bias
            out = (input - bias).div(log_scale.exp() + 1e-8)
            return out




    def init(self, data, init_scale=1.0):
        with torch.no_grad():
            # [channels, 1, 1]
            log_scale = self.log_scale
            bias = self.bias
            out = data * log_scale.exp() + bias
            out = out.transpose(0, 1).contiguous().view(self.in_channels, -1)
            # [n_channels, 1, 1]
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.data.copy_(inv_stdv.log())
            self.bias.data.copy_(-mean * inv_stdv)
            # return self.forward(data)


class MultiScalePrior(nn.Module):
    """
    Prior in multi-scale architecture
    """
    def __init__(self, in_channels, hidden_channels, h_channels, factor, transform, alpha,
                 coupling_type, h_type, activation, normalize, num_groups,use_1x1=False, condition_nice=False,
                 attention=False,heads=4,spatial_size=8,cond_conv=False,cond_conv_hidden_channels=None,p_dropout=0.):
        super().__init__()
        permutation = InvertibleConvLU1d if use_1x1 else Shuffle
        self.conv1x1 = permutation(in_channels)
        self.coupling = NICE2d(in_channels, hidden_channels=hidden_channels, h_channels=h_channels if condition_nice else 0,
                               transform=transform, alpha=alpha, factor=factor,
                               type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                               activation=activation, normalize=normalize, num_groups=num_groups,
                               attention=attention,heads=heads,spatial_size=spatial_size,
                               cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                               p_dropout=p_dropout
                               )
        out_channels = in_channels // factor
        self.z1_channels = self.coupling.z1_channels
        assert out_channels + self.z1_channels == in_channels
        self.actnorm = ActNorm2dFlow(out_channels)

    # def sync(self):
    #     self.conv1x1.sync()

    def forward(self, input: torch.Tensor, h=None, reverse=False):
        if not reverse:
            # conv1x1
            out, logdet_accum = self.conv1x1(input)
            # coupling
            out, logdet = self.coupling(out, h=h)
            logdet_accum = logdet_accum + logdet
            # actnorm
            out1, out2 = split2d(out, self.z1_channels)
            out2, logdet = self.actnorm(out2)
            logdet_accum = logdet_accum + logdet
            out = unsplit2d([out1, out2])
            return out, logdet_accum
        else:
            # actnorm
            out1, out2 = split2d(input, self.z1_channels)
            out2 = self.actnorm(out2,reverse=True)
            out = unsplit2d([out1, out2])
            # coupling
            out = self.coupling(out, h=h,reverse=True)

            # conv1x1
            out = self.conv1x1(out,reverse=True)

            return out


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

    # def logdet(self):
    #     val = torch.sum(self.log_s) * torch.ones(self.b, self.h, self.w).to(self.log_s.device)
    #     assert not torch.sum(torch.isnan(val)), "Nan occured in InvConv logdet {} {}".format(val, self.log_s)
    #     assert not torch.sum(torch.isinf(val)), "Inf occured in InvConv logdet {} {}".format(val, self.log_s)
    #     # val = val.view(1,1,1,1)
    #     return val

    def forward(self, x, reverse=False):
        self.b = x.size(0)
        self.h = x.size(2)
        self.w = x.size(3)
        if not reverse:
            wl = self.l * self.lmask + self.eye
            wu = self.u * self.umask + torch.diag(self.sign_s * torch.exp(self.log_s))
            weight = torch.matmul(self.permutated, torch.matmul(wl, wu)).view(self.nf, self.nf, 1, 1)

            logdet = torch.sum(self.log_s) * self.h * self.w * torch.ones(self.b,
                                                                          device=self.log_s.get_device() if self.log_s.get_device() >=0 else "cpu")
            assert not torch.sum(torch.isnan(logdet)), "Nan occured in InvConv logdet {} {}".format(logdet, self.log_s)
            assert not torch.sum(torch.isinf(logdet)), "Inf occured in InvConv logdet {} {}".format(logdet, self.log_s)
            return F.conv2d(x, weight), logdet
        else:
            wl = self.l * self.lmask + self.eye
            wu = self.u * self.umask + torch.diag(self.sign_s * torch.exp(self.log_s))
            weight = torch.matmul(torch.inverse(wu), torch.matmul(torch.inverse(wl),
                                                                  torch.inverse(self.permutated))).view(self.nf,
                                                                                                        self.nf, 1,
                                                                                                        1)

            return F.conv2d(x, weight)


class MultiscaleStack(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        levels = self.config["levels"]
        factors = self.config["factors"]
        self.reshape = self.config['reshape'] if 'reshape' in self.config else 'none'



        h_channels = self.config['h_channels'] if 'h_channels' in self.config else 0
        condition_nice = 'condition_nice' in self.config and self.config['condition_nice']

        assert len(levels) == len(factors)
        assert self.reshape in ['none', 'up', 'down']

        if self.reshape != 'none':
            self.reshape_transform = Reshape() if self.reshape == 'down' else Inverse(Reshape())

            if h_channels > 0:
                self.h_transforms = nn.ModuleList()

            self.reshape_step = len(levels) // 2

        self.blocks = nn.ModuleList()
        in_channels = self.config['flow_in_channels']
        mid_channels = in_channels * self.config['flow_mid_channels_factor']

        for i, (steps, f) in enumerate(zip(levels,factors)):
            assert isinstance(steps,list)
            if h_channels > 0 and self.reshape != 'none' and i >= self.reshape_step:
                act_hT = Conv2dTransposeBlock(h_channels,h_channels,3,2,1,norm='group') if self.reshape=='up' else Conv2dBlock(h_channels,h_channels,3,2,1,norm='group')
                self.h_transforms.append(act_hT)
            if self.reshape != 'none' and i == self.reshape_step:
                in_channels = in_channels*4 if self.reshape == 'down' else in_channels//4
                mid_channels = in_channels * self.config['flow_mid_channels_factor']
            self.blocks.append(MultiScaleInternal(MaCowStep, num_steps=steps,
                                       in_channels=in_channels,
                                       hidden_channels=mid_channels, h_channels=h_channels,
                                       factor=f, transform=self.config["transform"],
                                       prior_transform=self.config["prior_transform"],
                                       kernel_size=self.config["kernel_size"],
                                       coupling_type=self.config["coupling_type"], activation=self.config["activation"],
                                       use_1x1=self.config["use1x1"] if "use1x1" in self.config else False,
                                       condition_nice=condition_nice))


    def forward(self,x,xc=None,reverse=False):
        xc_in = xc
        if not reverse:
            logdet_accum = x.new_zeros(x.size(0))
            for i,flow in enumerate(self.blocks):
                if i == self.reshape_step:
                    x, _ = self.reshape_transform(x)

                if i >= self.reshape_step:
                    xc_in = self.h_transforms[i-self.reshape_step](xc)
                x, logdet = flow(x,xc_in)
                logdet_accum = logdet_accum + logdet

            return x, logdet_accum
        else:
            for i,flow in enumerate(reversed(self.blocks)):

                if i < self.reshape_step:
                    xc_in = self.h_transforms[-1-i](xc)
                else:
                    xc_in = xc

                if i == self.reshape_step:
                    x, _ = self.reshape_transform(x,reverse=True)

                x = flow(x,xc_in,reverse=True)



            return x

class MultiscaleMixCDF(nn.Module):

    def __init__(self, num_steps, dimension, hidden_channels_factor, h_channels,
                 factor=2, heads=4, components=4,  prior_transform='affine', alpha=1.0,
                 coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, use_1x1=False):
        super().__init__()
        self.reshape="none"
        num_layers = len(num_steps)
        assert num_layers < factor
        self.layers = nn.ModuleList()
        self.priors = nn.ModuleList()
        self.shuffle_layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        #self.nonlinear = InvLeakyRelu()
        in_channels = dimension[0]
        self.reshape = 'none'

        channel_step = in_channels // factor
        for num_step in num_steps:
            dimension[0] = int(in_channels/2)
            hidden_channels = hidden_channels_factor * in_channels
            norm = [ActNorm(in_channels,logdet=True) for _ in range(num_step)]
            self.norms.append(nn.ModuleList(norm))
            layer = [UnconditionalMixCDFCouplingBlock(dimension,hidden_channels,2,heads,components) for _ in range(num_step)]
            self.layers.append(nn.ModuleList(layer))
            prior = MultiScalePrior(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                                    transform=prior_transform, alpha=alpha, factor=factor,
                                    coupling_type=coupling_type, h_type=h_type, activation=activation,
                                    normalize=normalize, num_groups=num_groups)
            self.priors.append(prior)
            permutation = InvertibleConvLU1d if use_1x1 else Shuffle
            self.shuffle_layers.append(permutation(in_channels))
            in_channels = in_channels - channel_step
            assert in_channels == prior.z1_channels
            factor = factor - 1
        self.z_channels = in_channels
        assert len(self.layers) == len(self.priors)


    def forward(self, input: torch.Tensor, h=None, reverse=False):
        if not reverse:
            out = input
            # [batch]
            logdet_accum = input.new_zeros(input.size(0))
            outputs = []
            for norm, layer, prior, shuffle in zip(self.norms,self.layers, self.priors, self.shuffle_layers):
                # out, logdet = self.nonlinear(out)
                # logdet_accum = logdet_accum + logdet
                for norm_step,step in zip(norm,layer):
                    out , logdet = norm_step(out)
                    logdet_accum = logdet_accum + logdet
                    out, logdet = step(out, h=h)
                    logdet_accum = logdet_accum + logdet
                out, logdet = prior(out, h=h)
                logdet_accum = logdet_accum + logdet
                out, logdet = shuffle(out)
                logdet_accum = logdet_accum + logdet
                # split
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            outputs.append(out)
            outputs.reverse()
            out = unsplit2d(outputs)
            return out, logdet_accum
        else:
            out = input
            outputs = []
            for prior in self.priors:
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            # [batch]
            for norm, layer, prior,shuffle in zip(reversed(self.norms), reversed(self.layers), reversed(self.priors),reversed(self.shuffle_layers)):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
                out = shuffle(out, reverse=True)
                out = prior(out, h=h,reverse=True)
                for norm_step,step in zip(reversed(norm), reversed(layer)):
                    out = step(out, h=h,reverse=True)
                    out = norm_step(out,reverse=True)
                #out = self.nonlinear(out,reverse=True)

            assert len(outputs) == 0
            return out


class MultiScaleInternal(nn.Module):
    """
    Multi-scale architecture internal block.
    """
    def __init__(self, flow_step, num_steps, in_channels, hidden_channels, h_channels,
                 factor=2, transform='affine', prior_transform='affine', alpha=1.0,
                 kernel_size=(2, 3), coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, use_1x1=False, condition_nice=False,
                 attention=False,heads=4,spatial_size=8,cond_conv=False,cond_conv_hidden_channels=None, p_dropout=0.):
        super().__init__()
        self.reshape = 'none'
        assert self.reshape in ['none','up','down']
        num_layers = len(num_steps)
        assert num_layers < factor
        self.layers = nn.ModuleList()
        self.priors = nn.ModuleList()
        self.shuffle_layers = nn.ModuleList()
        # self.nonlinear = InvLeakyRelu()


        channel_step = in_channels // factor
        for num_step in num_steps:


            layer = [flow_step(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                               transform=transform, alpha=alpha, coupling_type=coupling_type, h_type=h_type,
                               activation=activation, normalize=normalize, num_groups=num_groups,
                               kernel_size=kernel_size,
                               condition_nice=condition_nice, attention=attention, heads=heads,
                               spatial_size=spatial_size,cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                               p_dropout=p_dropout)
                     for _ in range(num_step)]
            self.layers.append(nn.ModuleList(layer))
            prior = MultiScalePrior(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                                    transform=prior_transform, alpha=alpha, factor=factor,
                                    coupling_type=coupling_type, h_type=h_type, activation=activation,
                                    normalize=normalize, num_groups=num_groups, condition_nice=condition_nice,
                                    attention=attention, heads=heads, spatial_size=spatial_size,
                                    cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                    p_dropout=p_dropout)
            self.priors.append(prior)
            permutation = InvertibleConvLU1d if use_1x1 else Shuffle
            self.shuffle_layers.append(permutation(in_channels))
            in_channels = in_channels - channel_step


            assert in_channels == prior.z1_channels
            factor = factor - 1
        self.z_channels = in_channels
        assert len(self.layers) == len(self.priors)


    def forward(self, input: torch.Tensor, h=None, reverse=False):
        if not reverse:
            out = input
            h_in = h
            # [batch]
            logdet_accum = input.new_zeros(input.size(0))
            outputs = []
            for i ,(layer, prior, shuffle) in enumerate(zip(self.layers, self.priors, self.shuffle_layers)):
                # out, logdet = self.nonlinear(out)
                # logdet_accum = logdet_accum + logdet


                for step in layer:
                    out, logdet = step(out, h=h_in)
                    logdet_accum = logdet_accum + logdet
                out, logdet = prior(out, h=h_in)
                logdet_accum = logdet_accum + logdet
                out, logdet = shuffle(out)
                logdet_accum = logdet_accum + logdet
                # split
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            outputs.append(out)
            outputs.reverse()
            out = unsplit2d(outputs)
            return out, logdet_accum
        else:
            out = input
            outputs = []
            for prior in self.priors:
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            # [batch]
            for layer, prior,shuffle in zip(reversed(self.layers), reversed(self.priors),reversed(self.shuffle_layers)):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
                out = shuffle(out, reverse=True)
                out = prior(out, h=h,reverse=True)
                for step in reversed(layer):
                    out = step(out, h=h,reverse=True)
                #out = self.nonlinear(out,reverse=True)

            assert len(outputs) == 0
            return out




class MaCowUnit(nn.Module):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """

    def __init__(self, in_channels, kernel_size, h_channels=0,
                 transform='affine', alpha=1.0, h_type=None, activation='relu',
                 cond_conv=False,cond_conv_hidden_channels=None, p_dropout=0.):
        super().__init__()
        self.cond_conv = cond_conv and h_channels > 0
        if  self.cond_conv:
            assert cond_conv_hidden_channels is not None
            self.cond_conv_block = GatedConv2d(dim=h_channels)
        self.conv1 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), order='A',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation,
                                    p_dropout=p_dropout)
        self.conv2 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), order='B',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation,
                                    p_dropout=p_dropout)
        self.actnorm1 = ActNorm2dFlow(in_channels)

        self.conv3 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), order='C',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation,
                                    p_dropout=p_dropout)
        self.conv4 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), order='D',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation,p_dropout=p_dropout)
        self.actnorm2 = ActNorm2dFlow(in_channels)

    def forward(self, input: torch.Tensor, h=None, reverse=False):

        if self.cond_conv:
            h = self.cond_conv_block(h)

        if not reverse:
            # MCF1
            out, logdet_accum = self.conv1(input, h=h)
            # MCF2
            out, logdet = self.conv2(out, h=h)
            logdet_accum = logdet_accum + logdet
            # ActNorm1
            out, logdet = self.actnorm1(out)
            logdet_accum = logdet_accum + logdet
            # MCF3
            out, logdet = self.conv3(out, h=h)
            logdet_accum = logdet_accum + logdet
            # MCF4
            out, logdet = self.conv4(out, h=h)
            logdet_accum = logdet_accum + logdet
            # ActNorm2
            out, logdet = self.actnorm2(out)
            logdet_accum = logdet_accum + logdet
            return out, logdet_accum
        else:
            out = self.actnorm2(input,reverse=True)
            # MCF4
            out = self.conv4(out, h=h,reverse=True)
            # MCF3
            out = self.conv3(out, h=h,reverse=True)
            # ActNorm1
            out = self.actnorm1(out,reverse=True)

            # MCF2
            out = self.conv2(out, h=h,reverse=True)

            # MCF1
            out = self.conv1(out, h=h,reverse=True)
            return out



class MaCowStep(nn.Module):
    """
        A step of Macow Flows
        """

    def __init__(self, in_channels, kernel_size, hidden_channels, h_channels,
                 transform='affine', alpha=1.0, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, **kwargs):
        super().__init__()
        num_units = 2
        condition_nice = kwargs['condition_nice']
        attention = 'attention' in kwargs and kwargs['attention']
        heads = kwargs['heads'] if 'heads' in kwargs else 4
        ssize = kwargs['spatial_size'] if 'spatial_size' in kwargs else 8
        cond_conv = kwargs['cond_conv'] if 'cond_conv' in kwargs else False
        cond_conv_hidden_channels = kwargs['cond_conv_hidden_channels'] if 'cond_conv_hidden_channels' in kwargs else None
        p_dropout=kwargs['p_dropout'] if 'p_dropout' in kwargs else 0.
        self.actnorm1 = ActNorm2dFlow(in_channels)
        self.conv1x1 = Shuffle(in_channels)
        units = [MaCowUnit(in_channels, kernel_size, h_channels=h_channels, transform=transform,
                           alpha=alpha, h_type=h_type, activation=activation,
                           cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                           p_dropout=p_dropout)
                 for _ in range(num_units)]
        self.units1 = nn.ModuleList(units)
        self.coupling1_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels if condition_nice else 0, transform=transform, alpha=alpha,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups,
                                   attention=attention,heads=heads,spatial_size=ssize,
                                   cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                   p_dropout=p_dropout)
        self.coupling1_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels if condition_nice else 0, transform=transform, alpha=alpha,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups,
                                   attention=attention,heads=heads,spatial_size=ssize,
                                   cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                   p_dropout=p_dropout)

        self.actnorm2 = ActNorm2dFlow(in_channels)

        units = [MaCowUnit(in_channels, kernel_size, h_channels=h_channels, transform=transform,
                           alpha=alpha, h_type=h_type, activation=activation,
                           cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                           p_dropout=p_dropout)
                 for _ in range(num_units)]
        self.units2 = nn.ModuleList(units)
        self.coupling2_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels if condition_nice else 0, transform=transform, alpha=alpha,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups,
                                   attention=attention,heads=heads,spatial_size=ssize,
                                   cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                   p_dropout=p_dropout)
        self.coupling2_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels if condition_nice else 0, transform=transform, alpha=alpha,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups,
                                   attention=attention,heads=heads,spatial_size=ssize,
                                   cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                   p_dropout=p_dropout)
    #
    # def sync(self):
    #     self.conv1x1.sync()


    def forward(self, input: torch.Tensor, h=None,reverse=False):
        if not reverse:
            # part1
            out, logdet_accum = self.actnorm1(input)
            out, logdet = self.conv1x1(out)
            logdet_accum = logdet_accum + logdet
            for unit in self.units1:
                out, logdet = unit(out, h=h)
                logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling1_up(out, h=h)
            logdet_accum = logdet_accum + logdet
            out, logdet = self.coupling1_dn(out, h=h)
            logdet_accum = logdet_accum + logdet
            # part 2
            out, logdet = self.actnorm2(out)
            logdet_accum = logdet_accum + logdet
            for unit in self.units2:
                out, logdet = unit(out, h=h)
                logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling2_up(out, h=h)
            logdet_accum = logdet_accum + logdet
            out, logdet = self.coupling2_dn(out, h=h)
            logdet_accum = logdet_accum + logdet
            return out, logdet_accum
        else:
            out = self.coupling2_dn(input, h=h,reverse=True)
            out = self.coupling2_up(out, h=h,reverse=True)


            for unit in reversed(self.units2):
                out = unit(out, h=h,reverse=True)


            out = self.actnorm2(out,reverse=True)

            # part1
            out = self.coupling1_dn(out, h=h,reverse=True)
            out = self.coupling1_up(out, h=h,reverse=True)


            for unit in reversed(self.units1):
                out = unit(out, h=h,reverse=True)


            out = self.conv1x1(out,reverse=True)
            # logdet_accum = logdet_accum + logdet

            out = self.actnorm1(out,reverse=True)
            # logdet_accum = logdet_accum + logdet
            return out#, logdet_accum

class HierarchicalConvCouplingBlock(nn.Module):
    """
    Prior in multi-scale architecture
    """
    def __init__(self, channels, hidden_channels, h_channels, transform, alpha,
                 coupling_type, h_type, activation, normalize, num_groups,use_1x1=False,
                 attention=False,heads=4,spatial_size=8,n_blocks=2):
        super().__init__()
        permutation = InvertibleConvLU1d if use_1x1 else Shuffle
        self.shuffle = permutation(channels)
        self.norm_layer = ActNorm2dFlow(channels)
        blocks = [NICE2d(channels, hidden_channels=hidden_channels, h_channels=h_channels,
                               transform=transform, alpha=alpha,
                               type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                               activation=activation, normalize=normalize, num_groups=num_groups,
                               attention=attention,heads=heads,spatial_size=spatial_size) for _ in range(n_blocks)]

        self.coupling = nn.ModuleList(blocks)


    def forward(self,x,xc=None,reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            for b in self.coupling:
                h, ld = b(h,xc)
                logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            for b in reversed(self.coupling):
                h = b(h, xc, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    # def forward(self, input: torch.Tensor, h=None, reverse=False):
    #     if not reverse:
    #         # conv1x1
    #         out, logdet_accum = self.conv1x1(input)
    #         # coupling
    #         out, logdet = self.coupling(out, h=h)
    #         logdet_accum = logdet_accum + logdet
    #         # actnorm
    #         out1, out2 = split2d(out, self.z1_channels)
    #         out2, logdet = self.conv1x1(out2)
    #         logdet_accum = logdet_accum + logdet
    #         out = unsplit2d([out1, out2])
    #         return out, logdet_accum
    #     else:
    #         # actnorm
    #         out1, out2 = split2d(input, self.z1_channels)
    #         out2 = self.actnorm(out2,reverse=True)
    #         out = unsplit2d([out1, out2])
    #         # coupling
    #         out = self.coupling(out, h=h,reverse=True)
    #
    #         # conv1x1
    #         out = self.conv1x1(out,reverse=True)
    #
    #         return out


class HierarchicalConvCouplingFlow(nn.Module):

    def __init__(self, num_steps, in_channels, hidden_channels_factor, h_channels,
                 factor=2, transform='affine', prior_transform='affine', alpha=1.0, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, use_1x1=False, condition_nice=False,
                 attention=False,heads=4,spatial_size=8, n_blocks=2):
        super().__init__()
        self.reshape = "none"
        num_layers = len(num_steps)
        assert num_layers < factor
        self.layers = nn.ModuleList()
        self.priors = nn.ModuleList()
        self.shuffle_layers = nn.ModuleList()
        # self.nonlinear = InvLeakyRelu()

        channel_step = in_channels // factor
        for num_step in num_steps:
            hidden_channels = hidden_channels_factor*in_channels
            layer = [HierarchicalConvCouplingBlock(in_channels,hidden_channels,h_channels,transform,alpha,
                                                   coupling_type,h_type,activation,normalize,num_groups,
                                                   attention=attention,heads=heads,spatial_size=spatial_size,n_blocks=n_blocks) for _ in range(num_step)]
            self.layers.append(nn.ModuleList(layer))
            prior = MultiScalePrior(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                                    transform=prior_transform, alpha=alpha, factor=factor,
                                    coupling_type=coupling_type, h_type=h_type, activation=activation,
                                    normalize=normalize, num_groups=num_groups, condition_nice=condition_nice,
                                    attention=attention, heads=heads, spatial_size=spatial_size)
            self.priors.append(prior)
            permutation = InvertibleConvLU1d if use_1x1 else Shuffle
            self.shuffle_layers.append(permutation(in_channels))
            in_channels = in_channels - channel_step
            assert in_channels == prior.z1_channels
            factor = factor - 1
        self.z_channels = in_channels
        assert len(self.layers) == len(self.priors)
        assert len(self.shuffle_layers) == len(self.layers)

    def forward(self, input: torch.Tensor, h=None, reverse=False):
        if not reverse:
            out = input
            # [batch]
            logdet_accum = input.new_zeros(input.size(0))
            outputs = []
            for layer, prior, shuffle in zip(self.layers, self.priors, self.shuffle_layers):
                # out, logdet = self.nonlinear(out)
                # logdet_accum = logdet_accum + logdet
                for step in layer:
                    out, logdet = step(out, xc=h)
                    logdet_accum = logdet_accum + logdet
                out, logdet = prior(out, h=h)
                logdet_accum = logdet_accum + logdet
                out, logdet = shuffle(out)
                logdet_accum = logdet_accum + logdet
                # split
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            outputs.append(out)
            outputs.reverse()
            out = unsplit2d(outputs)
            return out, logdet_accum
        else:
            out = input
            outputs = []
            for prior in self.priors:
                out1, out2 = split2d(out, prior.z1_channels)
                outputs.append(out2)
                out = out1

            # [batch]
            for layer, prior, shuffle in zip(reversed(self.layers), reversed(self.priors),
                                             reversed(self.shuffle_layers)):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
                out = shuffle(out, reverse=True)
                out = prior(out, h=h, reverse=True)
                for step in reversed(layer):
                    out = step(out, xc=h, reverse=True)
                # out = self.nonlinear(out,reverse=True)

            assert len(outputs) == 0
            return out
