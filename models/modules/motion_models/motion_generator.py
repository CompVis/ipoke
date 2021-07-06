import torch.nn as nn, torch
import torch.nn.functional as F
from models.modules.autoencoders.util import Spade, ADAIN, Norm3D
from torch.nn.utils import spectral_norm


class generator_block(nn.Module):

    def __init__(self, n_in, n_out, pars):
        super().__init__()
        self.learned_shortcut = (n_in != n_out)
        n_middle = min(n_in, n_out)

        self.conv_0 = nn.Conv3d(n_in, n_middle, 3, 1, 1)
        self.conv_1 = nn.Conv3d(n_middle, n_out, 3, 1, 1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(n_in, n_out, 1, bias=False)

        if pars['spectral_norm']:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = Spade(n_in, pars) if pars['CN_content'] == 'spade' else Norm3D(n_in, pars)
        self.norm_1 = ADAIN(n_middle, pars) if pars['CN_motion'] == 'ADAIN' else Norm3D(n_middle, pars)

        if self.learned_shortcut:
            self.norm_s = Norm3D(n_in, pars)

    def forward(self, x, cond1, cond2):

        x_s = self.shortcut(x)

        dx = self.conv_0(self.actvn(self.norm_0(x, cond2)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, cond1)))

        out = x_s + dx

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Generator(nn.Module):
    def __init__(self, dic):
        super().__init__()

        self.img_size = dic["img_size"]
        nf = dic['decoder_factor']
        self.z_dim = dic['z_dim']
        self.fmap_start = 16*nf

        self.fc = nn.Linear(dic['z_dim'], 4*4*16*nf)
        self.head_0 = generator_block(16*nf, 16*nf, dic)

        self.g_0 = generator_block(16*nf, 16*nf, dic)
        self.g_1 = generator_block(16*nf, 8*nf, dic)
        self.g_2 = generator_block(8*nf, 4*nf, dic)
        self.g_3 = generator_block(4*nf, 2*nf, dic)
        self.g_4 = generator_block(2*nf, 1*nf, dic)

        self.conv_img = nn.Conv3d(nf, 3, 3, padding=1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=0.02)
            # nn.init.orthogonal_(m.weight.data, gain=0.02)
            if not isinstance(m.bias, type(None)):
                nn.init.constant_(m.bias.data, 0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, motion):

        x = self.fc(motion).reshape(img.size(0), -1, 1, 4, 4)
        # x = torch.ones(img.size(0), self.fmap_start, 1, 4, 4).cuda()

        x = self.head_0(x, motion, img)

        x = F.interpolate(x, scale_factor=2)
        x = self.g_0(x, motion, img)

        x = F.interpolate(x, scale_factor=2)
        x = self.g_1(x, motion, img)

        x = F.interpolate(x, scale_factor=2)
        x = self.g_2(x, motion, img)

        x = F.interpolate(x, scale_factor=(2, 2, 2))
        x = self.g_3(x, motion, img)

        if self.img_size > 64:
            x = F.interpolate(x, scale_factor=(1, 2, 2))
        x = self.g_4(x, motion, img)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x.transpose(1, 2)

