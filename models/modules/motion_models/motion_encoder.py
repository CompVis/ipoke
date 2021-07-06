import torch.nn as nn, torch
import numpy as np


######################################################################################################
###3D-ConvNet Implementation from https://github.com/tomrunia/PyTorchConv3D ##########################

def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet18_alternative(**kwargs):
    model = ResNetMotionEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model



def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, dic):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.spatial_size = dic['img_size']
        channels = dic['ENC_M_channels']
        # currently no  deterministic motion encoder required
        self.be_determinstic = False
        self.conv1  = nn.Conv3d(3, channels[0], kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1    = nn.GroupNorm(num_groups=16, num_channels=channels[0])
        self.relu   = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[1], layers[0])
        self.layer2 = self._make_layer(block, channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[3], layers[2], stride=2)
        last_channels = channels[3]
        if self.spatial_size // 2**3 > 4:
            self.layer4 = self._make_layer(block, channels[4], layers[3], stride=2)
            last_channels = channels[4]
        if self.spatial_size // 2**4 > 4:
            self.layer5 = self._make_layer(block, channels[5], layers[3], stride=2)
            last_channels = channels[5]

        self.conv_mu = nn.Conv2d(last_channels, dic['z_dim'], 4, 1, 0)
        self.conv_var = nn.Conv2d(last_channels, dic['z_dim'], 4, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu(emb).reshape(emb.size(0), -1), self.conv_var(emb).reshape(emb.size(0), -1)
        eps = torch.FloatTensor(logvar.size()).normal_().cuda()
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.spatial_size // 2 ** 3 > 4:
            x = self.layer4(x)
        if self.spatial_size // 2 ** 4 > 4:
            x = self.layer5(x)
        return self.reparameterize(x.squeeze(2))



class ResNetMotionEncoder(nn.Module):

    def __init__(self, block, layers, dic):
        super().__init__()
        self.be_determinstic = 'deterministic' in dic and dic['deterministic']
        channels = dic['ENC_M_channels']
        self.inplanes = channels[0]
        self.spatial_size = dic['img_size']
        max_frames = dic['max_frames']
        self.min_ssize = dic['min_spatial_size'] if 'min_spatial_size' in dic else 8

        self.conv1  = nn.Conv3d(3, channels[0], kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1    = nn.GroupNorm(num_groups=16, num_channels=channels[0])
        self.relu   = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        test = np.log2(max_frames)
        first_block_down = len(channels)-1 < int(np.ceil(test)) or dic['full_seq']
        stride1 = (2,1,1) if first_block_down else 1
        self.layer1 = self._make_layer(block, channels[1], layers[0],stride=stride1)
        self.layer2 = self._make_layer(block, channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[3], layers[2], stride=2)
        last_channels = channels[3]

        self.stride4 = (2,1,1) if dic['full_seq'] and max_frames >= 16 else None

        if self.spatial_size // 2**3 > self.min_ssize:
                self.stride4 = 2


        if self.stride4 is not None:
            if len(channels)<5:
                channels.append(channels[-1])
                print(f"Warning: adding one additional layer to motion encoder with channels={channels[-1]}")
            self.layer4 = self._make_layer(block, channels[4], layers[3], stride=self.stride4)
            last_channels = channels[4]
        if self.spatial_size // 2**4 > self.min_ssize:
            self.layer5 = self._make_layer(block, channels[5], layers[3], stride=2)
            last_channels = channels[5]


        self.conv_mu = nn.Conv2d(last_channels, dic['z_dim'], 3, 1, 1)
        self.conv_var = nn.Conv2d(last_channels, dic['z_dim'], 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, emb):
        mu, logvar = self.conv_mu(emb), self.conv_var(emb)
        eps = torch.FloatTensor(logvar.size()).normal_().cuda()
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu), mu, logvar

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.stride4 is not None:
            x = self.layer4(x)
        if self.spatial_size // 2 ** 4 > self.min_ssize:
            x = self.layer5(x)
        if self.be_determinstic:
            _, out, _ = self.reparameterize(x.squeeze(2))
            return out, out , out
        else:
            return self.reparameterize(x.squeeze(2))




if __name__ == '__main__':
    ## Test 3dconvnet with dummy input
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    config = {'ENC_M_channels': [32,64,128,128], 'z_dim': 64, 'img_size': 64, 'max_frames': 7}

    model = resnet18_alternative(dic=config ).cuda()

    print(f'model has {sum(p.numel() for p in model.parameters())} parameters')

    dummy = torch.rand((2, 3, config['max_frames'], 64, 64)).cuda()
    out, *_= model(dummy)
    print(out.shape)


