import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.init import _calculate_fan_in_and_fan_out, uniform_, normal_
import math
from torch.nn import functional as F
import numpy as np
import functools
from opt_einsum import contract


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):
    def forward(self, input):
        return input


class L2NormConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        init=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        # init
        if callable(init):
            self.init_fn = init
        else:
            self.init_fn = lambda: False
        normal_(self.weight, mean=0.0, std=0.05)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)

    def forward(self, x, it=None):
        W_norm = F.normalize(self.weight, dim=[1, 2, 3], p=2)
        x = F.conv2d(
            x, W_norm, self.bias, stride=self.stride, padding=self.padding
        )
        # training attribute is inherited from nn.Module
        if self.init_fn() and self.training:
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], keepdim=True)
            self.gamma.data = 1.0 / torch.sqrt(var + 1e-10)
            self.beta.data = -mean * self.gamma

        return self.gamma * x + self.beta


class LayerNormConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        x = self.conv(x)
        return self.norm(x)


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


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):
        return self.down(x)


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False,
                 out_dim=None, data_init=False, last_zero=False):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        module = CustomLinear if data_init else nn.Linear
        layers.append(module(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(module(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        if data_init and last_zero:
            layers.append(
                module(hidden_dim, dim if out_dim is None else out_dim,zero_init=True))
        else:
            layers.append(
                module(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BasicResNet(nn.Module):

    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False,
                 out_dim=None, data_init=False, last_zero=False):
        super().__init__()
        layers = []
        winit = 'data' if data_init else 'xavier'
        # initialization scheme for last layer depends on last_zero-parameter
        linit = 'zeros' if last_zero else winit

        layers.append(GatedConv2d(dim,dim_out=hidden_dim,winit=winit))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(GatedConv2d(hidden_dim, dim_out=hidden_dim,winit=winit))
            layers.append(nn.LeakyReLU())

        layers.append(GatedConv2d(hidden_dim, dim_out=out_dim,winit=linit))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, zero_init=False):
        super().__init__(in_features,out_features,bias)
        self.register_buffer('initialized',torch.tensor(1 if zero_init else 0,dtype=torch.uint8))
        if zero_init:
            self.weight.data = torch.zeros(self.weight.shape)
            self.bias.data = torch.zeros(self.bias.shape)

    def initialize(self,x):
        with torch.no_grad():
            out = super().forward(x)
            out = out.transpose(0, 1).contiguous().view(self.out_features, -1)
            # [n_out_features]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = 1. / (std + 1e-6)

            self.weight.mul_(inv_stdv.view(self.out_features,1))
            if self.bias is not None:
                self.bias.add_(-mean).mul_(inv_stdv)

    def forward(self, x):
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)
        return super().forward(x)

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
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
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).type_as(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


_norm_options = {
        "in": nn.InstanceNorm2d,
        "bn": nn.BatchNorm2d,
        "an": ActNorm}

class GINActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
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
            alpha = torch.prod(std + 1e-6)
            self.scale.data.copy_(alpha / (std + 1e-6))

    def get_scale(self):
        scale = self.scale[:, :-1, :, :]
        lastscale = 1.0 / (torch.prod(scale) + 1e-6)
        lastscale = lastscale * torch.ones(1, 1, 1, 1).to(lastscale)
        scale = torch.cat((scale, lastscale), dim=1)
        return scale

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.get_scale() * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            logdet = torch.zeros(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.get_scale() - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class GIN2ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
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
            alpha = torch.prod(std + 1e-6)
            self.scale.data.copy_(alpha / (std + 1e-6))

    def get_scale(self):
        scale = self.scale
        totalscale = torch.prod(scale, dim=1, keepdim=True)
        scale = scale / (
                    totalscale + 1e-6)  # TODO this might be an issue scale -> 0
        return scale

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.get_scale() * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            # TODO better return real logdet?
            logdet = torch.zeros(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.get_scale() - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


# karpathy's made + conditioning


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return nn.functional.linear(input, self.mask * self.weight, self.bias)


class ARFullyConnectedNet(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1,
                 natural_ordering=False, ncond=0):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        self.ncond = ncond

        # define a simple MLP neural net
        self.net = nn.ModuleList()
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.append(MaskedLinear(h0, h1))

        if self.ncond > 0:
            self.condnet = nn.ModuleList()
            hs = [ncond] + hidden_sizes + [nout]
            for h0, h1 in zip(hs, hs[1:]):
                self.condnet.append(MaskedLinear(h0, h1))

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(
            self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1,
                                    size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        for l, m in zip(self.net, masks):
            l.set_mask(m)

    def forward(self, x, y=None):
        assert len(x.shape) == 2
        assert x.shape[1] == self.nin
        if self.ncond > 0:
            assert y is not None
            assert len(y.shape) == 2
            assert y.shape[1] == self.ncond
            assert y.shape[0] == x.shape[0]
            for i in range(len(self.net)):
                if i > 0:
                    x = nn.functional.relu(x)
                    y = nn.functional.relu(y)
                y = self.condnet[i](y)
                x = self.net[i](x) + y
            return x
        else:
            assert y is None
            for i in range(len(self.net)):
                if i > 0:
                    x = nn.functional.relu(x)
                x = self.net[i](x)
            return x


class BasicUnConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False,
                 out_dim=None):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        assert self.out_dim % self.dim == 0
        self.factor = self.out_dim // self.dim

        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=hidden_dim,
                                kernel_size=1))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Conv1d(in_channels=hidden_dim,
                                    out_channels=hidden_dim, kernel_size=1))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv1d(in_channels=hidden_dim,
                                out_channels=self.factor, kernel_size=1))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) == 2
        xs = x.shape
        x = x[:, None, :]  # (bs,1,dim)
        x = self.main(x)  # (bs, out_dim, dim)
        x = x.reshape(-1, self.out_dim)
        return x


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='AN', width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        self.wm = width_multiplier
        if in_channels is None:
            self.in_channels = int(self.wm*64*min(2**(self.scale-1), 16))
        else:
            self.in_channels = in_channels
        self.out_channels = int(self.wm*64*min(2**self.scale, 16))
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=self.out_channels),
                Activate()])


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None,
                 width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.wm = width_multiplier
        self.in_channels = int(self.wm*64*min(2**(self.scale-1), 16))
        if in_channels is not None:
            print('Warning: Ignoring `scale` parameter in DenseEncoderLayer due to given number of input channels.')
            self.in_channels = in_channels
        self.out_channels = out_size
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])



# convolutional flow blocks


class LNorm(nn.Module):

    def __init__(self,dim,axes:list=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)
        if axes:
            assert len(axes) == 3
            self.axes = axes
        else:
            self.axes = 1

    def forward(self,x,eps=1e-5):
        mu = torch.mean(x,dim=self.axes,keepdim=True)
        s = (x-mu).pow(2).mean(self.axes)[:,None]
        x = (x - mu) / torch.sqrt(s + eps)
        return self.scale * x + self.shift


class NIN(nn.Module):

    def __init__(self,in_units,out_units,initializer="data"):
        super().__init__()
        assert initializer in ["data", "xavier", "zeros"]
        self.out_units = out_units
        self.b = nn.Parameter(torch.zeros(out_units),requires_grad=True)
        self.W = nn.Parameter(torch.zeros((in_units,out_units)) if initializer=="zeros" else
                              torch.randn((in_units,out_units)), requires_grad=True)

        if initializer =="xavier":
            nn.init.xavier_normal_(self.W.data)
            self.register_buffer('initialized', torch.tensor(1, dtype=torch.uint8))
        else:
            self.register_buffer('initialized', torch.tensor(1, dtype=torch.uint8)
            if initializer=="zeros" else torch.tensor(0, dtype=torch.uint8))

    def _initialize(self,input):
        y = torch.matmul(input,self.W)
        var,mean = torch.var_mean(y,dim=0)
        scale_init = 1. / torch.sqrt(var + 1e-8)
        self.W.data.copy_(self.W.data*scale_init[None])
        self.b.data.copy_(-mean * scale_init)


    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        sh = x.shape
        x = x.reshape((np.prod(sh[:-1]), -1))
        if self.initialized.item() == 0:
            self._initialize(x)
            self.initialized.fill_(1)
        # make channels last

        out= torch.matmul(x,self.W) + self.b
        out = out.reshape(list(sh[:-1])+[self.out_units]).permute(0,3,1,2)
        return out

class ConcatELU(nn.Module):
    def __init__(self,axis):
        super().__init__()
        self.axis=axis
        self.elu = nn.ELU()


    def forward(self,x):
        return self.elu(torch.cat([x,-x],dim=self.axis))


class Gate(nn.Module):
    def __init__(self,axis):
        super().__init__()
        self.axis=axis
        self.act = nn.Sigmoid()

    def forward(self,x):
        a, b = torch.chunk(x,2,dim=self.axis)
        return a * self.act(b)

class Conv2d(nn.Module):
    """
    Conv Layer with data dependent initialization
    """
    def __init__(self,in_dim,out_dim,ks=3,s=1,p=1, initializer="data"):
        super().__init__()
        assert initializer in ["data","xavier","zeros"]
        self.weight = nn.Parameter(torch.zeros((out_dim,in_dim,ks,ks)) if initializer == "zeros" else torch.randn((out_dim,in_dim,ks,ks))
                                   ,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_dim),requires_grad=True)

        self.stride = s
        self.pad = p
        if initializer == "xavier":
            nn.init.xavier_normal_(self.weight.data)
            #nn.init.xavier_normal_(self.bias.data)
            self.register_buffer('initialized', torch.tensor(1,dtype=torch.uint8))
        else:
            self.register_buffer('initialized', torch.tensor(1 if initializer=="zeros" else 0, dtype=torch.uint8))

    def _initialize(self,input):
        # init without bias
        y = F.conv2d(input,self.weight,stride=self.stride,padding=self.pad)
        var,mean = torch.var_mean(y,dim=[0,2,3])
        scale_init = 1. / torch.sqrt(var + 1e-8)
        self.weight.data.copy_(self.weight.data * scale_init[:,None,None,None])
        self.bias.data.copy_(-mean*scale_init)

    def forward(self,x):
        if self.initialized.item() == 0:
            self._initialize(x)
            self.initialized.fill_(1)

        return F.conv2d(x,self.weight,self.bias,stride=self.stride,padding=self.pad)


class GatedConv2d(nn.Module):

    def __init__(self,dim,conditional=False,dim_cond=None,actfn=ConcatELU(axis=1),p_dropout=0.,winit="data",dim_out=None):
        super().__init__()
        self.conditional = conditional


        if dim_out is None:
            dim_out = dim
            self.adapt_cn = False
        else:
            self.adapt_cn = True
            self.conv_sc = Conv2d(dim,dim_out,1,1,p=0,initializer=winit)

        dim_multiplier = 2 if isinstance(actfn,ConcatELU) else 1

        self.conv1 = Conv2d(dim_multiplier*dim,dim,3,1,1,initializer=winit)
        self.conv2 = Conv2d(dim_multiplier*dim,dim_out*2,3,1,1,initializer=winit)

        self.nonlinear = actfn
        self.gate = Gate(axis=1)

        if conditional:
            assert dim_cond is not None
            self.cond_conv = Conv2d(dim_multiplier*dim_cond,dim,3,1,1,initializer=winit)

        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None



    def forward(self,x,xc=None):
        c1 = self.conv1(self.nonlinear(x))
        if self.adapt_cn:
            x = self.conv_sc(x)
        if self.conditional:
            c1 += self.cond_conv(self.nonlinear(xc))
        c1 = self.nonlinear(c1)
        if self.dropout:
            c1 = self.dropout(c1)

        c2 = self.conv2(c1)

        return x + self.gate(c2)


class GatedAttentionLayer(nn.Module):

    def __init__(self,dimension,heads,p_dropout=0.,winit="data"):
        super().__init__()
        self.channels, self.H, self.W = dimension
        self.timesteps = self.H * self.W
        assert self.channels % heads == 0
        self.heads = heads
        # per head dimension
        self.dim = self.channels // self.heads

        self.proj_1 = NIN(self.channels,3*self.channels,initializer=winit)
        self.proj_2 = NIN(self.channels,2*self.channels,initializer=winit)


        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None

        self.softm = nn.Softmax(-1)
        self.gate = Gate(1)

    def forward(self,x,pos_emb):
        b, c, h, w = x.shape
        x_ = x + pos_emb[None]
        # b,c, h, w --> b, 3*c, h, w
        x_ = self.proj_1(x_)
        # b, 3*c, h, w --> 3, b, heads, timesteps, per-head dim
        x_ = x_.reshape(b,3,self.heads,self.dim,self.timesteps).permute(1,0,2,4,3)
        # generate keys queries and values
        Q,K,V = torch.unbind(x_,dim=0)
        #weighting
        dev = Q.get_device() if Q.is_cuda else "cpu"
        W = torch.matmul(Q,K.transpose(-2,-1)) / torch.full((1,1,1,1),np.sqrt(float(self.dim)),device=dev)
        W = self.softm(W)
        A = torch.matmul(W,V)
        # b, heads, timesteps, per-head dim --> b, timesteps, heads,  per-head dim
        A = A.permute(0,1,3,2)
        A = A.reshape(b,c,self.timesteps)
        # back to original dimensions
        gate_in = A.reshape(b,c,h,w)
        if self.dropout:
            gate_in = self.dropout(gate_in)

        gate_in = self.proj_2(gate_in)
        return x + self.gate(gate_in)

class BasicConvAttnBlock(nn.Module):

    def __init__(self,dimension,heads,cond=False,c_cond=None,p_dropout=0.,winit="data"):
        super().__init__()
        c,h,w = dimension
        self.gated_resnet = GatedConv2d(dim=c,conditional=cond,dim_cond=c_cond,p_dropout=p_dropout,winit=winit)
        self.attn = GatedAttentionLayer(dimension,heads,p_dropout,winit=winit)
        #self.norm = LNorm(c)
        self.norm = nn.LayerNorm(dimension)
    def forward(self,x,pos_emb,xc=None):
        x = self.gated_resnet(x,xc)
        x = self.norm(x)
        x = self.attn(x,pos_emb)
        return self.norm(x)

class MixCDFParameterTemplate(nn.Module):

    def __init__(self,in_dim,hidden_dim,blocks,heads,components,cond=False,cond_channels=None,p_dropout=0.,winit="data"):
        super().__init__()

        self.c,self.h,self.w = in_dim
        self.components = components
        hidden_shape = (hidden_dim,self.h,self.w)
        # pos embedding for attention layers
        self.pos_emb = nn.Parameter(torch.randn(hidden_shape), requires_grad=True)

        self.conv_in = Conv2d(self.c,hidden_dim,3,1,1,initializer=winit)

        self.blocks = nn.ModuleList()
        for n in range(blocks):
            self.blocks.append(BasicConvAttnBlock(hidden_shape,heads,cond,
                                                  cond_channels,p_dropout,winit=winit))

        self.nonlinearity = ConcatELU(1)
        self.conv_out = Conv2d(2*hidden_dim,self.c * (2+3*components),3,1,1,initializer="zeros")
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU()

    def forward(self,x,xc=None):

        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x,self.pos_emb)

        x = self.nonlinearity(x)
        x = self.conv_out(x)
        x = x.reshape((x.size(0),self.c,2+3*self.components,self.h,self.w))
        #x = self.lrelu(x)
        # shape : b, c, h, w
        #s, t = - self.lrelu(x[:, :, 0]), x[:, :, 1]
        s,t = self.tanh(x[:,:,0]), x[:,:,1]

        # shape: b,c,components, h, w
        ml_logits, ml_means, ml_logscales = torch.chunk(x[:,:,2:],3,dim=2)
        ml_logscales = torch.maximum(ml_logscales,torch.full_like(ml_logscales,-7.))

        return s,t,ml_logits,ml_means, ml_logscales

class MixCDFParameterTemplate2(nn.Module):

    def __init__(self,in_dim,hidden_dim,blocks,heads,components,cond=False,cond_channels=None,p_dropout=0.):
        super().__init__()

        self.c,self.h,self.w = in_dim
        self.components = components
        hidden_shape = (hidden_dim,self.h,self.w)
        # pos embedding for attention layers
        self.pos_emb = nn.Parameter(torch.randn(hidden_shape), requires_grad=True)

        self.conv_in = Conv2d(self.c,hidden_dim,3,1,1)

        self.blocks = nn.ModuleList()
        for n in range(blocks):
            self.blocks.append(BasicConvAttnBlock(hidden_shape,heads,cond,
                                                  cond_channels,p_dropout))

        self.nonlinearity = ConcatELU(1)
        self.conv_out = Conv2d(2*hidden_dim,self.c * (2+3*components),3,1,1)
        self.tanh = nn.Tanh()

    def forward(self,x,xc=None):

        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x,self.pos_emb)

        x = self.nonlinearity(x)
        x = self.conv_out(x)
        x = x.permute(0,2,3,1)
        x = x.reshape((x.size(0),self.h,self.w,self.c,2+3*self.components))

        # shape : b, c, h, w
        s,t = self.tanh(x[:,:,:,:,0]).permute(0,3,1,2), x[:,:,:,:,1].permute(0,3,1,2)

        # shape: b,c,components, h, w
        ml_logits, ml_means, ml_logscales = torch.chunk(x[:,:,:,:,2:],3,dim=4)
        ml_logscales = torch.maximum(ml_logscales,torch.full_like(ml_logscales,-7.))

        return s,t,ml_logits,ml_means, ml_logscales


class LogisticLogCDF(nn.Module):

    def __init__(self):
        super().__init__()
        self.logsigm = nn.LogSigmoid()

    def forward(self,x,mean,logscale):
        z = (x-mean)*torch.exp(-logscale)
        return self.logsigm(z)

class LogisticLogPDF(nn.Module):

    def __init__(self):
        super().__init__()
        self.softp = nn.Softplus()

    def forward(self,x,mean,logscale):
        z = (x-mean)*torch.exp(-logscale)
        return z - logscale - 2* self.softp(z)





class MixLogCDF(nn.Module):

    def __init__(self,dim=2):
        super().__init__()
        self.logcdf = LogisticLogCDF()
        self.dim = dim
        self.logsoft = nn.LogSoftmax(dim=self.dim)


    def forward(self,x,prior_logits,means,logscales):
        prior = self.logsoft(prior_logits)

        sumexp = prior + self.logcdf(x.unsqueeze(self.dim),means,logscales)
        return torch.logsumexp(sumexp,dim=self.dim)


class MixLogPDF(nn.Module):
    def __init__(self,dim=2):
        super().__init__()
        self.logpdf = LogisticLogPDF()
        self.dim = dim
        self.logsoft = nn.LogSoftmax(dim=self.dim)

    def forward(self,x,prior_logits,means,logscales,exp=True):
        prior = self.logsoft(prior_logits)

        sumexp = prior + self.logpdf(x.unsqueeze(self.dim),means,logscales)
        log_mix_log_cdf = torch.logsumexp(sumexp,self.dim)
        if exp:
            return torch.exp(log_mix_log_cdf)
        else:
            return log_mix_log_cdf

def inv_mixlogcdf(y,prior_logits, means, logscales, tol=1e-10,max_iter=500,dim=2):
    """
    inverse cumulative mixlog function via bisection
    :param y:
    :param prior_logits:
    :param means:
    :param logscales:
    :param tol:
    :param mix_iter:
    :return:
    """
    #device = y.get_device() if y.is_cuda else "cpu"
    assert torch.all(y<1.) and torch.all(y>0.)
    x_ = torch.zeros_like(y)

    y = torch.clone(y)

    maxscales = torch.sum(logscales.exp(),dim=dim,keepdim=True)
    lb_ = torch.min(means - 50 * maxscales,dim=dim)[0]
    ub_ = torch.max(means + 50 * maxscales, dim=dim)[0]
    #diff_ = torch.tensor(np.inf,device=device)

    def bisect(x,lb,ub):
        cur_y = torch.exp(MixLogCDF(dim=dim)(x,prior_logits,means,logscales))
        gt = torch.greater(cur_y,y).to(y.dtype)
        lt = torch.logical_not(gt.to(torch.bool)).to(y.dtype)
        new_x = gt * (x + lb) / 2. + lt * (x + ub) / 2.
        new_lb = gt * lb + lt * x
        new_ub = gt * x + lt * ub
        diff = torch.max(torch.abs(new_x-x))
        return new_x,new_lb,new_ub,diff

    for it in range(max_iter):
        x_,lb_,ub_,diff_ = bisect(x_,lb_,ub_)
        if diff_ < tol:
            break

    assert x_.shape == y.shape
    return x_


#macow related stuff
class NICEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, s_channels, dilation):
        super(NICEConvBlock, self).__init__()
        self.conv1 = Conv2dWeightNorm(in_channels + s_channels, hidden_channels, kernel_size=3, dilation=dilation, padding=dilation, bias=True)
        self.conv2 = Conv2dWeightNorm(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.conv3 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation, bias=True,init_zeros=True)
        self.activation = nn.ELU()

    # def init(self, x, s=None, init_scale=1.0):
    #     if s is not None:
    #         x = torch.cat([x, s], dim=1)
    #
    #     out = self.activation(self.conv1.init(x, init_scale=init_scale))
    #
    #     out = self.activation(self.conv2.init(out, init_scale=init_scale))
    #
    #     out = self.conv3.init(out, init_scale=0.0)
    #
    #     return out

    def forward(self, x, s=None):
        if s is not None:
            x = torch.cat([x, s], dim=1)

        out = self.activation(self.conv1(x))

        out = self.activation(self.conv2(out))

        out = self.conv3(out)
        return out

class MultiHeadAttention2d(nn.Module):
    def __init__(self, channels, heads, dropout=0.0):
        super(MultiHeadAttention2d, self).__init__()
        self.proj = Conv2dWeightNorm(channels, 3 * channels, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        assert channels % heads == 0
        self.features = channels
        self.heads = heads

    def forward(self, x, pos_enc=None):
        # [batch, channels, height, width]
        if pos_enc is not None:
            x = x + pos_enc
        bs, channels, height, width = x.size()
        heads = self.heads
        dim = channels // heads
        # [batch, 3 * channels, height, width]
        c = self.proj(x)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        #attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', (queries, keys)).div(math.sqrt(dim))
        attn_weights = contract('bhdij,bhdkl->bhijkl', queries, keys,backend="torch").div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        #out = torch.einsum('bhdt,bhijt->bhdij', (values, attn_weights))
        out = contract('bhdt,bhijt->bhdij', values, attn_weights,backend="torch")
        if self.dropout is not None:
            out = self.dropout(out)
        # merge heads
        # [batch, channels, heads, dim]
        out = x + out.view(bs, channels, height, width)
        return out



class SelfAttnLayer(nn.Module):
    def __init__(self, channels, heads, dropout=0.0):
        super(SelfAttnLayer, self).__init__()
        self.attn = MultiHeadAttention2d(channels, heads, dropout=dropout)
        self.gn = nn.GroupNorm(heads, channels)

    def forward(self, x, pos_enc=None):
        return self.gn(self.attn(x, pos_enc=pos_enc))

class Conv2dWeightNorm(nn.Module):
    """
    Conv2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, init_zeros=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        if init_zeros:
            nn.init.constant_(self.conv.weight, 0.)
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)
        else:
            self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(1 if not init_zeros else 1, dtype=torch.uint8))

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

            self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.add_(-mean).mul_(inv_stdv)

    def forward(self, input):
        if self.initialized.item() == 0:
            with torch.no_grad():
                self.init(input)
                self.initialized.fill_(1)

        return self.conv(input)


class NICESelfAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, s_channels, slice, heads, pos_enc=True, dropout=0.0):
        super(NICESelfAttnBlock, self).__init__()
        self.nin1 = NIN2d(in_channels + s_channels, hidden_channels, bias=True)
        self.attn = SelfAttnLayer(hidden_channels, heads, dropout=dropout)
        self.nin2 = NIN4d(hidden_channels, hidden_channels, bias=True)
        self.activation = nn.ELU()
        self.nin3 = NIN2d(hidden_channels, out_channels, bias=True)
        self.slice_height, self.slice_width = slice
        # positional enc
        if pos_enc:
            self.register_buffer('pos_enc', torch.zeros(hidden_channels, self.slice_height, self.slice_width))
            pos_enc = np.array([[[(h * self.slice_width + w) / np.power(10000, 2.0 * (i // 2) / hidden_channels)
                                  for i in range(hidden_channels)]
                                 for w in range(self.slice_width)]
                                for h in range(self.slice_height)])
            pos_enc[:, :, 0::2] = np.sin(pos_enc[:, :, 0::2])
            pos_enc[:, :, 1::2] = np.cos(pos_enc[:, :, 1::2])
            pos_enc = np.transpose(pos_enc, (2, 0, 1))
            self.pos_enc.copy_(torch.from_numpy(pos_enc).float())
        else:
            self.register_buffer('pos_enc', None)

    def forward(self, x, s=None):
        # [batch, in+s, height, width]
        bs, _, height, width = x.size()
        if s is not None:
            x = torch.cat([x, s], dim=1)

        # slice2d
        # [batch*fh*fw, hidden, slice_heigth, slice_width]
        x = self.slice2d(x, self.slice_height, self.slice_width, init=False)
        x = self.attn(x, pos_enc=self.pos_enc)

        # unslice2d
        # [batch, hidden, height, width]
        x = self.unslice2d(x, height, width, init=False)
        # compute output
        # [batch, out, height, width]
        out = self.nin3(x)
        return out

    def slice2d(self, x: torch.Tensor, slice_height, slice_width, init, init_scale=1.0) -> torch.Tensor:
        batch, n_channels, height, width = x.size()
        assert height % slice_height == 0 and width % slice_width == 0
        fh = height // slice_height
        fw = width // slice_width

        # [batch, channels, height, width] -> [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = x.view(-1, n_channels, fh, slice_height, fw, slice_width)
        # [batch, channels, factor_height, slice_height, factor_width, slice_width] -> [batch, factor_height, factor_width, channels, slice_height, slice_width]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # [batch, factor_height, factor_width, hidden, slice_height, slice_width]
        x = self.nin1(x)
        # [batch * factor_height * factor_width, hidden, slice_height, slice_width]
        hidden_channels = x.size(3)
        x = x.view(-1, hidden_channels, slice_height, slice_width)
        return x

    def unslice2d(self, x: torch.Tensor, height, width, init, init_scale=1.0) -> torch.Tensor:
        _, n_channels, slice_height, slice_width = x.size()
        assert height % slice_height == 0 and width % slice_width == 0
        fh = height // slice_height
        fw = width // slice_width

        # [batch, factor_height, factor_width, channels, slice_height, slice_width]
        x = x.view(-1, fh, fw, n_channels, slice_height, slice_width)
        # [batch, factor_height, factor_width, channels, slice_height, slice_width] -> [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = x.permute(0, 3, 1, 4, 2, 5)
        # [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = self.nin2(x)
        x = self.activation(x)
        # [batch, channels, height, width]
        x = x.view(-1, n_channels, height, width)
        return x


def norm(p: torch.Tensor, dim: int):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return norm(p.transpose(0, dim), 0).transpose(0, dim)


class NIN2d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NIN2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = nn.Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))


    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_g.data.copy_(norm(self.weight_v, 0))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def compute_weight(self):
        return self.weight_v * (self.weight_g / norm(self.weight_v, 0))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.init(input)
            self.initialized.fill_(1)

        weight = self.compute_weight()
        #out = torch.einsum('...cij,oc->...oij', (input, weight))
        out = contract('...cij,oc->...oij', input, weight,backend="torch")
        if self.bias is not None:
            out = out + self.bias
        return out


    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            weight = self.compute_weight()
            # fixme test the follwowing line and check speedup (if there's any):
            #out = contract('...cij,oc->...oij', x, weight,backend="torch")
            #out = torch.einsum('...cij,oc->...oij', (x, weight))
            out = contract('...cij,oc->...oij', x, weight,backend="torch")
            if self.bias is not None:
                out = out + self.bias
            out_features, height, width = out.size()[-3:]
            assert out_features == self.out_features
            # [batch, out_features, h * w] - > [batch, h * w, out_features]
            out = out.view(-1, out_features, height * width).transpose(1, 2)
            # [batch*height*width, out_features]
            out = out.contiguous().view(-1, out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.bias is not None:
                mean = mean.view(out_features, 1, 1)
                inv_stdv = inv_stdv.view(out_features, 1, 1)
                self.bias.add_(-mean).mul_(inv_stdv)



class NIN4d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NIN4d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = nn.Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, 1, 1, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_g.data.copy_(norm(self.weight_v, 0))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def compute_weight(self):
        return self.weight_v * (self.weight_g / norm(self.weight_v, 0))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.init(input)
            self.initiliazed.fill_(1)

        weight = self.compute_weight()
        #out = torch.einsum('bc...,oc->bo...', (input, weight))
        out = contract('bc...,oc->bo...', input, weight,backend="torch")
        if self.bias is not None:
            out = out + self.bias
        return out


    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            weight = self.compute_weight()
            #out = torch.einsum('bc...,oc->bo...', (x, weight))
            out = contract('bc...,oc->bo...', x, weight,backend="torch")
            if self.bias is not None:
                out = out + self.bias
            batch, out_features = out.size()[:2]
            assert out_features == self.out_features
            # [batch, out_features, h * w] - > [batch, h * w, out_features]
            out = out.view(batch, out_features, -1).transpose(1, 2)
            # [batch*height*width, out_features]
            out = out.contiguous().view(-1, out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.bias is not None:
                mean = mean.view(out_features, 1, 1, 1, 1)
                inv_stdv = inv_stdv.view(out_features, 1, 1, 1, 1)
                self.bias.add_(-mean).mul_(inv_stdv)


class ShiftedConv2d(Conv2dWeightNorm):
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
            raise ValueError('unknown order: {}'.format(order))

        super(ShiftedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=0,
                                            stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, shifted=True):
        if shifted:
            input = F.pad(input, self.shift_padding)
            bs, channels, height, width = input.size()
            t, b, l, r = self.cut
            input = input[:, :, t:height + b, l:width + r]
        return self.conv(input)


####tests###

def test_logmix():
    import scipy.stats as ss

    n = 100
    xs = np.linspace(-1, 1, n)
    prior_logits = [.1, .2, 4., 8.]
    means = [-1., 0., 1., 5.]
    logscales = [-5., 0., 0.2, 0.3]

    mixlogcdf = MixLogCDF(dim=-1)
    mixlogpdf = MixLogPDF(dim=-1)

    prior_probs = np.exp(prior_logits) / np.exp(prior_logits).sum()
    scipy_probs = 0.
    scipy_cdfs = 0.
    for p, m, ls in zip(prior_probs, means, logscales):
        scipy_probs += p * ss.logistic.pdf(xs, m, np.exp(ls))
        scipy_cdfs += p * ss.logistic.cdf(xs, m, np.exp(ls))

    prior_logits = torch.stack([torch.tensor(prior_logits)] * n, dim=0)
    means = torch.stack([torch.tensor(means)] * n, dim=0)
    logscales = torch.stack([torch.tensor(logscales)] * n, dim=0)

    logpdfs = mixlogpdf(torch.from_numpy(xs),prior_logits,means,logscales,exp=False).numpy()
    logcdfs = mixlogcdf(torch.from_numpy(xs), prior_logits, means, logscales).numpy()

    assert scipy_probs.shape == logpdfs.shape
    assert np.allclose(logpdfs, np.log(scipy_probs))
    assert np.allclose(logcdfs, np.log(scipy_cdfs))
    print("passed")

def test_mixlogistic_invcdf():

    n = 64
    xs = torch.linspace(-1, 1, n)
    prior_logits = torch.stack([torch.tensor([.1, .2, 4])] * n, dim=0)
    means = torch.stack([torch.tensor([-1., 0., 1])] * n, dim=0)
    logscales = torch.stack([torch.tensor([-5., 0., 0.2])] * n, dim=0)



    logcdfs =torch.exp(MixLogCDF(dim=-1)(xs,prior_logits,means,logscales))
    inv_logcdfs = inv_mixlogcdf(logcdfs,prior_logits,means,logscales,dim=-1,tol=1e-15)

    assert inv_logcdfs.shape == xs.shape
    err = torch.max(torch.abs(inv_logcdfs - xs))
    print(err)
    assert err < 1e-5
    print('ok')

def test_space_to_depth():
    t = torch.randn((2,128,8,8))
    reshape = Reshape(2)
    t_r, _ = reshape(t)
    print(t_r.shape)
    t_out = reshape(t_r,reverse=True)

    assert torch.all(torch.eq(t,t_out))


if __name__ == '__main__':

    test_logmix()
    test_mixlogistic_invcdf()
    test_space_to_depth()