import torch
from torch import nn
import torchvision
from torchvision import models as tvmodels

from models.modules.autoencoders.util import ActNorm
from models.modules.autoencoders.distributions import DiagonalGaussianDistribution
from models.modules.autoencoders.biggan import load_variable_latsize_generator



class BigAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.be_deterministic = config['deterministic']
        if "n_out_channels" not in config:
            config.update({"n_out_channels": 3})
        self.encoder = ResnetEncoder(config)
        self.decoder = BigGANDecoderWrapper(config=config)

    def encode(self, input):
        h = input
        h = self.encoder(h)
        return DiagonalGaussianDistribution(h, deterministic=self.be_deterministic)

    def decode(self, input):
        h = input
        h = self.decoder(h.squeeze(-1).squeeze(-1))
        return h

    def forward(self, input):
        p = self.encode(input)
        img = self.decode(p.mode())
        return img, p.mode(), p

    def get_last_layer(self):
        return getattr(self.decoder.decoder.colorize.module, 'weight_bar')


class ClassUp(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_sigmoid=False, out_dim=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x.squeeze(-1).squeeze(-1))
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class BigGANDecoderWrapper(nn.Module):
    """Wraps a BigGAN into our autoencoding framework"""
    def __init__(self, config):
        super().__init__()
        z_dim = config['z_dim']
        self.do_pre_processing = config['pre_process']
        image_size = config['in_size']
        use_actnorm = config['use_actnorm_in_dec']
        pretrained = config['pretrained']
        class_embedding_dim = 1000
        use_adain = config["use_adain"] if "use_adain" in config else False

        self.map_to_class_embedding = ClassUp(z_dim, depth=2, hidden_dim=2*class_embedding_dim,
                                              use_sigmoid=False, out_dim=class_embedding_dim)
        self.decoder = load_variable_latsize_generator(image_size, z_dim,
                                                       pretrained=pretrained,
                                                       use_actnorm=use_actnorm,
                                                       n_class=class_embedding_dim,
                                                       n_channels=config["n_out_channels"],
                                                       adain=use_adain)


    def forward(self, x, labels=None):
        emb = self.map_to_class_embedding(x)
        x = self.decoder(x, emb)
        return x

class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None,
                 width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.wm = width_multiplier
        self.in_channels = int(self.wm*64*min(2**(self.scale-1), 16))
        if in_channels is not None:
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


_norm_options = {
        "in": nn.InstanceNorm2d,
        "bn": nn.BatchNorm2d,
        "an": ActNorm}

rescale = lambda x: 0.5*(x+1)

class ResnetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        __possible_resnets = {
            'resnet18': tvmodels.resnet18,
            'resnet34': tvmodels.resnet34,
            'resnet50': tvmodels.resnet50,
            'resnet101': tvmodels.resnet101
        }
        self.config = config
        self.do_pre_processing = config['pre_process']
        self.in_channels = self.config["n_in_channels"] if "n_in_channels" in self.config else 3
        self.use_inconv = self.in_channels != 3
        if self.use_inconv:
            # input map
            self.in_conv = nn.Conv2d(in_channels=self.config["n_in_channels"],out_channels=3,kernel_size=1)
            assert not self.do_pre_processing

        z_dim = config['z_dim']
        ipt_size =config['in_size']
        type_ = config['type']
        load_pretrained = config['pretrained']
        norm_layer = _norm_options[config['norm']]

        self.type = type_
        self.z_dim = z_dim
        self.model = __possible_resnets[type_](pretrained=load_pretrained, norm_layer=norm_layer)

        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))]
                )

        size_pre_fc = self._get_spatial_size(ipt_size)
        assert size_pre_fc[2]==size_pre_fc[3], 'Output spatial size is not quadratic'
        spatial_size = size_pre_fc[2]
        num_channels_pre_fc = size_pre_fc[1]
        # replace last fc
        self.model.fc = DenseEncoderLayer(0,
                                          spatial_size=spatial_size,
                                          out_size=2*z_dim,
                                          in_channels=num_channels_pre_fc)

    def forward(self, x):
        if self.do_pre_processing:
            x = self._pre_process(x)
        features = self.features(x)
        encoding = self.model.fc(features)
        return encoding

    def features(self, x):
        if self.do_pre_processing:
            x = self._pre_process(x)
        if self.use_inconv:
            x = self.in_conv(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

    def post_features(self, x):
        x = self.model.fc(x)
        return x

    def _pre_process(self, x):
        x = self.image_transform(x)
        return x

    def _get_spatial_size(self, ipt_size):
        x = torch.randn(1,self.in_channels , ipt_size, ipt_size)
        return self.features(x).size()

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]