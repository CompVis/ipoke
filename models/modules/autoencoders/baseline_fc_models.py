import torch
from torch import nn
from torch.nn.utils import  spectral_norm
import numpy as np

from models.modules.autoencoders.util import Conv2dBlock, ResBlock, Spade, NormConv2d
from models.modules.autoencoders.fully_conv_models import ConvEncoder


class FirstStageFCWrapper(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = BaselineFCEncoder(config=self.config)
        # no spade layer as these are only for encoder training
        self.config['architecture']['dec_channels'] = [self.config['architecture']['nf_max']] + self.encoder.depths
        self.config['architecture']['spectral_norm'] = True
        self.config['architecture'].update({'z_dim': self.config['architecture']['nf_max']})
        self.decoder = BaselineFCGenerator(config=self.config['architecture'],use_spade=False)
        self.be_deterministic = True

    def forward(self,x):
        enc = self.encoder(x)
        return self.decoder([enc],None)

class BaselineFCEncoder(ConvEncoder):

    def __init__(self,config):
        self.config = config
        n_stages = int(
            np.log2(self.config["data"]["spatial_size"][0] // 4))
        nf_max = self.config['architecture']['nf_max']
        nf_in = self.config['architecture']['nf_in']
        #always determinstic
        self.deterministic = True
        super().__init__(nf_in,nf_max,n_stages,variational=not self.deterministic)

        self.make_fc = NormConv2d(nf_max,nf_max,4,padding=0)

    def forward(self, x):
        # onky use output as model is not varaitional
        out, *_ =super().forward(x,sample_prior=False)
        out = self.make_fc(out).squeeze(dim=-1).squeeze(dim=-1)
        return out




class BaselineFCGenerator(nn.Module):

    def __init__(self,config, use_spade=True):
        super().__init__()
        channels = config['dec_channels']
        snorm = config['spectral_norm']
        latent_dim = config['z_dim']
        nc_out = config['nc_out'] if 'nc_out' in config else 3
        self.use_spade = use_spade

        self.blocks = nn.ModuleList()
        self.spade_blocks = nn.ModuleList()
        self.first_conv_nf = channels[0]
        self.n_stages = len(channels)-1
        if snorm:
            self.start_block = spectral_norm(nn.Linear(in_features=latent_dim,out_features=self.first_conv_nf * 16,))
        else:
            self.start_block = nn.Linear(in_features=latent_dim,out_features=channels[0] * 16,)
        nf = 0
        for i, nf in enumerate(channels[1:]):
            n_out = nf

            nf_in_dec = channels[i]
            self.blocks.append(ResBlock(nf_in_dec, n_out, norm='none' if self.use_spade else 'group', upsampling=True, snorm=config['spectral_norm']))
            if self.use_spade:
                self.spade_blocks.append(Spade(n_out, config))

        self.out_conv = Conv2dBlock(nf, nc_out, 3, 1, 1, norm="none",
                                    activation="tanh")


    def forward(self,actual_frame,start_frame,del_shape=True):
        x = self.start_block(actual_frame.pop() if del_shape else actual_frame[-1])
        x = x.reshape(x.size(0),self.first_conv_nf,4,4)
        for n in range(self.n_stages):
            x = self.blocks[n](x)
            if self.use_spade:
                x = self.spade_blocks[n](x, start_frame)

        if del_shape:
            assert not actual_frame
        out = self.out_conv(x)
        return out

# class BaseLineFCEncoder(nn.Module):

