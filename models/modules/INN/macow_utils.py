import torch
import math
from torch import nn
from torch.nn import functional as F

from models.modules.INN.modules import BasicConvAttnBlock, GatedConv2d


class Transform():
    def calc_params(self, params):
        return params

    @staticmethod
    def fwd(z: torch.Tensor, params):
        raise NotImplementedError

    @staticmethod
    def bwd(z: torch.Tensor, params):
        raise NotImplementedError



class Additive(Transform):
    def __init__(self):
        super(Additive, self).__init__()

    @staticmethod
    def fwd(z: torch.Tensor, params):
        mu = params
        z = z + mu
        logdet = z.new_zeros(z.size(0))
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, params) :
        mu = params
        z = z - mu
        logdet = z.new_zeros(z.size(0))
        return z, logdet


class Affine(Transform):
    def __init__(self, dim, alpha):
        super(Affine, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.tanh = nn.Tanh()

    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = self.tanh(log_scale.mul(0.5)).mul(self.alpha).add(1.0)
        return mu, scale

    @staticmethod
    def fwd(z: torch.Tensor, params) :
        mu, scale = params
        z = scale * z + mu
        logdet = scale.log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, params):
        mu, scale = params
        z = (z - mu).div(scale + 1e-12)
        logdet = scale.log().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


class ReLU(Transform):
    def __init__(self, dim):
        super(ReLU, self).__init__()
        self.dim = dim
        self.tanh = nn.Tanh()

    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = self.tanh(log_scale)
        return mu, scale

    @staticmethod
    def fwd(z: torch.Tensor, params):
        mu, scale = params
        scale = scale * z.gt(0.0).type_as(z) + 1
        z = scale * z + mu
        logdet = scale.log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, params):
        mu, scale = params
        z = z - mu
        scale = scale * z.gt(0.0).type_as(z) + 1
        z = z.div(scale + 1e-12)
        logdet = scale.log().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2) - 1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))


class NLSQ(Transform):
    # A = 8 * math.sqrt(3) / 9 - 0.05  # 0.05 is a small number to prevent exactly 0 slope
    logA = math.log(8 * math.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope

    def __init__(self, dim):
        super(NLSQ, self).__init__()
        self.dim = dim

    def calc_params(self, params):
        a, logb, cprime, logd, g = params.chunk(5, dim=self.dim)

        # for stability
        logb = logb.mul(0.4)
        cprime = cprime.mul(0.3)
        logd = logd.mul(0.4)

        # b = logb.add_(2.0).sigmoid_()
        # d = logd.add_(2.0).sigmoid_()
        # c = (NLSQ.A * b / d).mul(cprime.tanh_())

        c = (NLSQ.logA + logb - logd).exp_().mul(cprime.tanh_())
        b = logb.exp()
        d = logd.exp()
        return a, b, c, d, g

    @staticmethod
    def fwd(z: torch.Tensor, params):
        a, b, c, d, g = params

        arg = (d * z).add(g)
        denom = arg.pow(2).add(1)
        c = c / denom
        z = b * z + a + c
        logdet = torch.log(b - 2 * c * d * arg / denom)
        logdet = logdet.view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, params):
        a, b, c, d, g = params

        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        g = g.double()
        z = z.double()

        aa = -b * d.pow(2)
        bb = (z - a) * d.pow(2) - 2 * b * d * g
        cc = (z - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (z - a) * (1 + g.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (27 * aa.pow(3))

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        z = t - bb / (3 * aa)
        arg = d * z + g
        denom = arg.pow(2) + 1
        logdet = torch.log(b - 2 * c * d * arg / denom.pow(2))

        z = z.float()
        logdet = logdet.float().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


class SymmELU(Transform):
    def __init__(self, dim):
        super(SymmELU, self).__init__()
        self.dim = dim
        self.tanh = nn.Tanh()

    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = self.tanh(log_scale.mul_(0.5))
        return mu, scale

    @staticmethod
    def fwd(z: torch.Tensor, params):
        mu, scale = params
        sgn = torch.sign(z)
        tmp = torch.exp(-torch.abs(z))
        z = z - sgn * scale * (tmp - 1.0) + mu
        logdet = (scale * tmp + 1).log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, params):
        mu, scale = params
        z = -torch.sign(z) * scale * (torch.exp(-torch.abs(z)) - 1.0) + mu
        return z, None


class Conv2dWeightNorm(nn.Module):
    """
    Conv2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, zero_init=False):
        super(Conv2dWeightNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.zero_init = zero_init

    def reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self.conv(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)

            self.conv.weight_g.data.copy_(inv_stdv.view(n_channels, 1, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.data.copy_(-mean*inv_stdv)
            # return self(x)

    def forward(self, input):
        if self.initialized.item() == 0:
            self.init(input,init_scale=0. if self.zero_init else 1.)
            self.initialized.fill_(1)
        return self.conv(input)

class NICEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, activation, normalize=None, num_groups=None, h_channels=0,
                 attention=False,spatial_size=8, heads=4,cond_conv=False,cond_conv_hidden_channels=None, p_dropout=0.):
        super(NICEConvBlock, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        assert normalize in [None, 'batch_norm', 'group_norm', 'instance_norm']
        self.cond = h_channels > 0
        self.attention = attention
        self.dp = nn.Dropout(p=p_dropout)
        if attention:
            dimension = (hidden_channels, spatial_size, spatial_size)
            self.pos_emb = nn.Parameter(torch.randn(dimension), requires_grad=True)
            self.in_resnet = GatedConv2d(in_channels,dim_out=hidden_channels)

            self.conv1 = BasicConvAttnBlock(dimension=dimension,heads=heads)
            self.conv2 = BasicConvAttnBlock(dimension=dimension,heads=heads)
        else:
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        # initialize last layer with zeros to stabilize training at the beginning
        self.cond_conv = cond_conv
        if self.cond:
            hidden_cn_h = hidden_channels + h_channels
            if self.cond_conv:
                assert cond_conv_hidden_channels is not None
                self.cond_conv_block = GatedConv2d(dim=cond_conv_hidden_channels)
        else:
            hidden_cn_h = hidden_channels
        self.conv3 = Conv2dWeightNorm(hidden_cn_h, out_channels, kernel_size=3, padding=1, bias=True,zero_init=True)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=1e-1)

        if normalize is None:
            self.norm1 = None
            self.norm2 = None
        elif normalize == 'batch_norm':
            self.norm1 = nn.BatchNorm2d(hidden_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)
        elif normalize == 'instance_norm':
            self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(hidden_channels, affine=True)
        else:
            self.norm1 = nn.GroupNorm(num_groups, hidden_channels, affine=True)
            self.norm2 = nn.GroupNorm(num_groups, hidden_channels, affine=True)
        self.reset_parameters()

    def reset_parameters(self):
        if self.norm1 is not None and self.norm1.affine:
            # norm 1
            nn.init.constant_(self.norm1.weight, 1.0)
            nn.init.constant_(self.norm1.bias, 0.0)
            # norm 2
            nn.init.constant_(self.norm2.weight, 1.0)
            nn.init.constant_(self.norm2.bias, 0.0)

    def forward(self, x, h=None):
        if self.attention:
            out = self.in_resnet(x)
            out = self.conv1(out,self.pos_emb)
        else:
            out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        # conv2
        if self.attention:
            out = self.conv2(out,self.pos_emb)
        else:
            out = self.conv2(out)
        out = self.dp(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        if h is not None and self.cond:
            if self.cond_conv:
                h = self.cond_conv_block(h)
            out = torch.cat([out,h],dim=1)
        out = self.activation(out)
        # conv3
        out = self.conv3(out)
        return out

    # def init(self, x, h=None, init_scale=1.0):
    #     with torch.no_grad():
    #         out = self.conv1(x)
    #         if self.norm1 is not None:
    #             out = self.norm1(out)
    #         out = self.activation(out)
    #         # init conv2
    #         out = self.conv2(out)
    #         if self.norm2 is not None:
    #             out = self.norm2(out)
    #         if h is not None:
    #             out = out + h
    #         out = self.activation(out)
    #         # init conv3
    #         out = self.conv3.init(out, init_scale=0.0 * init_scale)
    #         return out


class LocalLinearCondNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LocalLinearCondNet, self).__init__()
        padding = kernel_size // 2
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x=None):
        return self.net(h)


class GlobalLinearCondNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(GlobalLinearCondNet, self).__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, h, x=None):
        out = self.net(h)
        bs, fs = out.size()
        return out.view(bs, fs, 1, 1)


class GlobalAttnCondNet(nn.Module):
    def __init__(self, q_dim, k_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(q_dim, out_dim, bias=True)
        self.key_proj = nn.Conv2d(k_dim, out_dim, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # key proj
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0)
        # query proj
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0)

    def forward(self, h, x):
        # [batch, out_dim]
        h = self.query_proj(h)
        # [batch, out_dim, height, width]
        key = self.key_proj(x)
        bs, dim, height, width = key.size()
        # [batch, height, width]
        attn_weights = torch.einsum('bd,bdhw->bhw', h, key)
        attn_weights = F.softmax(attn_weights.view(bs, -1), dim=-1).view(bs, height, width)
        # [batch, out_dim, height, width]
        out = h.view(bs, dim, 1, 1) * attn_weights.unsqueeze(1)
        return out


class MCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, order, activation,h_channels=None,p_dropout=0.):
        super(MCFBlock, self).__init__()
        if h_channels is not None:
            in_channels2 = hidden_channels+h_channels
        else:
            in_channels2 = hidden_channels
        self.shift_conv = ShiftedConv2d(in_channels, hidden_channels, kernel_size, order=order, bias=False)
        assert activation in ['relu', 'elu', 'leaky_relu']

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=1e-1)
        self.conv1x1 = Conv2dWeightNorm(in_channels2, out_channels, kernel_size=1, bias=True,zero_init=True)
        self.dp = nn.Dropout(p=p_dropout)


    def forward(self, x, h=None, shifted=True):
        c = self.shift_conv(x, shifted=shifted)
        if h is not None:
            c = torch.cat([c,h],dim=1)
        c = self.activation(c)
        c = self.conv1x1(c)
        c = self.dp(c)
        return c

    # def init(self, x, h=None, init_scale=1.0):
    #     with torch.no_grad():
    #         c = self.shift_conv(x)
    #         if h is not None:
    #             c = c + h
    #         c = self.activation(c)
    #         c = self.conv1x1.init(c, init_scale=0.0 * init_scale)
    #         return c


class ShiftedConv2d(nn.Conv2d):
    """
    Conv2d with shift operation.
    A -> top
    B -> bottom
    C -> left
    D -> right
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=1, groups=1, bias=True, order='A'):
        assert len(stride) == 2
        assert len(kernel_size) == 2
        assert order in {'A', 'B', 'C', 'D'}, 'unknown order: {}'.format(order)
        if order in {'A', 'B'}:
            assert kernel_size[1] % 2 == 1, 'kernel width cannot be even number: {}'.format(kernel_size)
        else:
            assert kernel_size[0] % 2 == 1, 'kernel height cannot be even number: {}'.format(kernel_size)

        self.order = order
        if order == 'A':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, kernel_size[0], 0)
            # top, bottom, left, right
            self.cut = (0, -1, 0, 0)
        elif order == 'B':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, 0, kernel_size[0])
            # top, bottom, left, right
            self.cut = (1, 0, 0, 0)
        elif order == 'C':
            # left, right, top, bottom
            self.shift_padding = (kernel_size[1], 0, (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 0, -1)
        elif order == 'D':
            # left, right, top, bottom
            self.shift_padding = (0, kernel_size[1], (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 1, 0)
        else:
            self.shift_padding = None
            raise ValueError('unknown order: {}'.format(order))

        super(ShiftedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=0,
                                            stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, shifted=True):
        if shifted:
            input = F.pad(input, self.shift_padding)
            bs, channels, height, width = input.size()
            t, b, l, r = self.cut
            input = input[:, :, t:height + b, l:width + r]
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    # def extra_repr(self):
    #     s = super(ShiftedConv2d, self).extra_repr()
    #     s += ', order={order}'
    #     s += ', shift_padding={shift_padding}'
    #     s += ', cut={cut}'
    #     return s.format(**self.__dict__)