import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.modules.autoencoders.util import Conv2dBlock, ResBlock, AdaINLinear, NormConv2d, Spade


class FirstStageWrapper(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.be_deterministic = self.config["architecture"]["deterministic"]
        n_stages = int(np.log2(self.config["data"]["spatial_size"][0] // self.config["architecture"]["min_spatial_size"]))
        nf_in_enc = self.config["architecture"]["nf_in"]
        if "poke_and_image" in self.config["architecture"] and self.config["architecture"]["poke_and_image"]:
            nf_in_enc+=3
        self.encoder = ConvEncoder(nf_in=nf_in_enc, nf_max=self.config["architecture"]["nf_max"],
                                   n_stages=n_stages, variational=not self.be_deterministic)
        decoder_channels = [self.config["architecture"]["nf_max"]] + self.encoder.depths
        self.decoder = ConvDecoder(self.config["architecture"]["nf_max"], decoder_channels, out_channels=self.config["architecture"]["nf_in"])

    def forward(self,x):
        enc, *_ = self.encoder(x)
        return self.decoder([enc],del_shape=False)

class ConvEncoder(nn.Module):
    def __init__(self, nf_in, nf_max, n_stages, variational=False, norm_layer = "group", layers=None, spectral_norm=True):
        super().__init__()

        self.variational = variational
        self.depths = []

        act = "elu" #if self.variational else "relu"

        blocks = []
        bottleneck = []
        nf = 32 if layers is None else layers[0]
        blocks.append(
            Conv2dBlock(
                nf_in, nf, 3, 2, norm=norm_layer, activation=act, padding=1,snorm=spectral_norm
            )
        )
        self.depths.append(nf)
        n_stages = n_stages if layers is None else len(layers)
        for n in range(n_stages - 1):
            blocks.append(
                ResBlock(
                    nf,
                    min(nf * 2, nf_max) if layers is None else layers[n+1],
                    stride = 2,
                    norm=norm_layer,
                    activation=act,
                    snorm=spectral_norm
                )
            )
            nf = min(nf * 2, nf_max) if layers is None else layers[n+1]
            self.depths.insert(0,nf)

        self.nf_in_bn = nf
        bottleneck.append(ResBlock(nf, nf_max,activation=act, norm=norm_layer))
        # if layers is None:
        #     bottleneck.append(ResBlock(nf_max, nf_max,activation=act, norm=norm_layer))


        if self.variational:
            self.make_mu = NormConv2d(nf_max,nf_max,3, padding=1)
            self.make_sigma = NormConv2d(nf_max,nf_max,3, padding=1)
            self.squash = nn.Sigmoid()

        self.model = nn.Sequential(*blocks)
        self.bottleneck = nn.Sequential(*bottleneck)

    def forward(self, input, sample_prior=False):
        out = self.model(input)
        mean = out
        out = self.bottleneck(out)
        logstd = None
        if self.variational:
            mean = self.make_mu(out)
            # normalize sigma in between
            logstd = self.squash(self.make_sigma(out))
            if sample_prior:
                out = torch.randn_like(mean)
            else:
                out = self.reparametrize(mean,logstd)

        return out, mean, logstd

    def reparametrize(self,mean,logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mean

class ConvDecoder(nn.Module):
    """
    Fully convolutional decoder consisting of resnet blocks, with optional skip connections (default no skip connections; if these
    shall be used, set n_skip_stages > 0
    """
    def __init__(self,nf_in, in_channels, n_skip_stages=0, spectral_norm=True, norm_layer="group",layers=None,out_channels=3):
        super().__init__()
        self.n_stages = len(in_channels)-1
        self.n_skip_stages = n_skip_stages

        self.blocks = nn.ModuleList()

        nf = nf_in
        self.in_block = ResBlock(nf,in_channels[0], snorm=spectral_norm, norm=norm_layer)

        for i,nf in enumerate(in_channels[1:]):
            if layers is None:
                n_out = nf

            nf_in_dec = 2 * nf if i < self.n_skip_stages else in_channels[i]
            # if layers is not None:
            #     nf_in_dec = 2 * nf
            #     n_out = in_channels[i+1] if i < len(in_channels) -1 else nf
            self.blocks.append(ResBlock(nf_in_dec, n_out , norm=norm_layer, upsampling=True,snorm=spectral_norm))

        self.out_conv = Conv2dBlock(nf,out_channels,3,1,1,norm="none",activation="tanh" if out_channels==3 else "none")

    def forward(self,shape, del_shape=True):
        x = self.in_block(shape.pop() if del_shape else shape[-1])
        for n in range(self.n_stages):
            if n < self.n_skip_stages:
                x = torch.cat([x,shape.pop() if del_shape else shape[self.n_skip_stages-1-n]],1)
            x = self.blocks[n](x)

        if del_shape:
            assert not shape
        out = self.out_conv(x)
        return out

class SpadeCondConvDecoder(nn.Module):

    def __init__(self,config,stacked_input=False):
        super().__init__()

        in_channels = config['dec_channels']

        self.n_stages = len(in_channels) - 1
        self.n_skip_stages = config['n_skip_stages'] if 'n_skip_stages' in config else 0
        out_channels = config['out_channels'] if 'out_channels' in config else 3

        self.blocks = nn.ModuleList()
        self.spade_blocks = nn.ModuleList()


        nf = 2*config['z_dim'] if stacked_input else config['z_dim']
        self.in_block = ResBlock(nf, in_channels[0], snorm=config['spectral_norm'], norm=config['norm'])

        for i, nf in enumerate(in_channels[1:]):
            n_out = nf

            nf_in_dec = 2 * nf if i < self.n_skip_stages else in_channels[i]
            # if layers is not None:
            #     nf_in_dec = 2 * nf
            #     n_out = in_channels[i+1] if i < len(in_channels) -1 else nf
            self.blocks.append(ResBlock(nf_in_dec, n_out, norm='none', upsampling=True, snorm=config['spectral_norm']))
            self.spade_blocks.append(Spade(n_out,config))

        self.out_conv = Conv2dBlock(nf, out_channels, 3, 1, 1, norm="none",
                                    activation="tanh" if out_channels == 3 else "none")

    def forward(self, actual_frame ,start_frame, del_shape=True):
        x = self.in_block(actual_frame.pop() if del_shape else actual_frame[-1])
        for n in range(self.n_stages):
            if n < self.n_skip_stages:
                x = torch.cat([x, actual_frame.pop() if del_shape else actual_frame[self.n_skip_stages - 1 - n]], 1)
            x = self.blocks[n](x)
            x = self.spade_blocks[n](x,start_frame)

        if del_shape:
            assert not actual_frame
        out = self.out_conv(x)
        return out