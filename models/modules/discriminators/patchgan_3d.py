import torch.nn as nn, torch
import math
from torch.nn.utils import spectral_norm

######################################################################################################
###3D-ConvNet Implementation from https://github.com/tomrunia/PyTorchConv3D ##########################


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def conv3x3x3(in_planes, out_planes, stride=1, stride_t=1):
    # 3x3x3 convolution with padding
    return spectral_norm(nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=[stride_t, stride, stride],
        padding=[1, 1, 1],
        bias=False))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, stride_t=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, stride_t)
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


# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  spatial_size,
#                  sequence_length,
#                  config):
#         super(ResNet, self).__init__()
#         # spatial_size = config["spatial_size"]
#         self.inplanes = 64
#         self.bce_loss = config["bce_loss"]
#         self.gp_weight = config["gp_weight"]
#         min_spatial_size = int(spatial_size / 8)
#         #sample_duration = dic.Network['sequence_length']-1
#         self.max_channels = config["max_channels"] if "max_channels" in config else 256
#         self.conv1 = spectral_norm(nn.Conv3d(
#             3,
#             64,
#             kernel_size=(3, 7, 7),
#             stride=(1, 2, 2),
#             padding=(1, 3, 3),
#             bias=False))
#         self.gn1      = nn.GroupNorm(num_groups=16, num_channels=64)
#         self.relu     = nn.ReLU(inplace=True)
#         self.maxpool  = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
#
#         self.layers = nn.ModuleList()
#         self.patch_temp = config["patch_temp_disc"]
#         self.spatio_temporal = config["spatio_temporal"] if"spatio_temporal" in config else False
#         if self.patch_temp:
#             self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
#             self.layers.append(self._make_layer(block, 128, layers[1], stride=1, stride_t=1))
#             self.layers.append(self._make_layer(block, 128, layers[2], stride=2, stride_t=1))
#             self.layers.append(self._make_layer(block, 256, layers[3], stride=2, stride_t=1))
#             last_size = int(math.ceil(spatial_size / 16))
#             last_duration = 1
#             self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
#             self.fc = nn.Linear(256 * block.expansion, config["num_classes"], bias=False)
#         else:
#             spatial_size /= 2
#             self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
#             n_channels = 64
#
#             n = 0
#             while sequence_length > 1:
#                 blocks = layers[n] if n<sequence_length-1 else layers[-1]
#                 n_channels = min(2*n_channels,self.max_channels)
#                 stride = 1 if spatial_size <= min_spatial_size else 2
#                 spatial_size = int(spatial_size / stride)
#                 stride_t = 1 if self.spatio_temporal else (2 if sequence_length > 1 else 1)
#                 self.layers.append(self._make_layer(block,n_channels,blocks,stride=stride,stride_t=stride_t))
#                 sequence_length = int(math.ceil(sequence_length / 2))
#                 n += 1
#
#             self.final = nn.Conv2d(n_channels,1,3,padding=1)
#
#
#         print(f"Temporal discriminator has {len(self.layers)} layers")
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.orthogonal_(m.weight)
#     def _make_layer(self, block, planes, blocks, stride=1, stride_t=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion or stride_t != 1:
#             downsample = nn.Sequential(
#                 spectral_norm(nn.Conv3d(
#                     self.inplanes,
#                     planes * block.expansion,
#                     kernel_size=[3, 3, 3],
#                     stride=[stride_t, stride, stride],
#                     padding=[1, 1, 1],
#                     bias=False)),
#                 nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, stride_t, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)
#     def forward(self, x, cond=None):
#         out = []
#         x = self.conv1(x)
#         x = self.gn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         out.append(x)
#
#         for n in range(len(self.layers)):
#             x = self.layers[n](x)
#             out.append(x)
#
#
#         if self.patch_temp:
#             x1 = self.avgpool(x)
#             output = []
#             for i in range(x1.size(2)):
#                 output.append(self.fc(x1[:,:,i].reshape(x1.size(0), -1)))
#             return torch.cat(output, dim=1), out
#         else:
#             output = self.final(x.squeeze(2))
#             return output, out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 spatial_size,
                 sequence_length,
                 config):
        super().__init__()

        self.inplanes = 64
        sample_duration = sequence_length - 1
        self.bce_loss = config["bce_loss"]
        self.gp_weight = config["gp_weight"]
        min_spatial_size = int(spatial_size / 8)
        #sample_duration = dic.Network['sequence_length']-1
        # max_channels = config["max_channels"] if "max_channels" in config else 256
        num_classes = config["num_classes"]
        if config['patch_temp_disc']:
            stride_t = 1
        else:
            stride_t = 2
        self.conv1 = spectral_norm(nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False))
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, stride_t=stride_t)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, stride_t=stride_t)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, stride_t=stride_t)
        last_duration = 1
        last_size = int(math.ceil(spatial_size / 16))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.orthogonal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride=1, stride_t=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                spectral_norm(nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=[3, 3, 3],
                    stride=[stride_t, stride, stride],
                    padding=[1, 1, 1],
                    bias=False)),
                nn.GroupNorm(num_channels=planes * block.expansion, num_groups=16))

        layers = []
        layers.append(block(self.inplanes, planes, stride, stride_t, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        x1 = self.avgpool(x)
        output = []
        for i in range(x1.size(2)):
            output.append(self.fc(x1[:, :, i].reshape(x1.size(0), -1)))

        return torch.cat(output, dim=1), out




    def loss(self, pred, real):
        if self.bce_loss:
            # vanilla gan loss
            return self.bce(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))
        else:
            # hinge loss
            if real:
                l = torch.mean(torch.nn.ReLU()(1.0 - pred))
            else:
                l = torch.mean(torch.nn.ReLU()(1.0 + pred))
            return l

    def gp(self, pred_fake, x_fake):
        batch_size = x_fake.size(0)
        grad_dout = torch.autograd.grad(
            outputs=pred_fake.sum(), inputs=x_fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_fake.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def gp2(self, pred_fake, x_fake):
        batch_size = x_fake.size(0)
        grad_dout = torch.autograd.grad(
            outputs=pred_fake.sum(), inputs=x_fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_fake.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()


    def fmap_loss(self, fmap1, fmap2, loss="l1"):
        recp_loss = 0
        for idx in range(len(fmap1)):
            if loss == "l1":
                recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
            if loss == "l2":
                recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
        return recp_loss / len(fmap1)