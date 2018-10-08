from model import common

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return KHU2(args, dilated.dilated_conv)
    else:
        return KHU2(args)


class ResNeXtBottleneck(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality
    self.downsample=downsample
    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    #print(planes*4)

  def forward(self, x):
    residual = x
    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(bottleneck, inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(bottleneck, inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    if self.downsample is not None:
      residual = self.downsample(x)

    return (residual + bottleneck)



class KHU2(nn.Module):
    def __init__(self, args, conv=common.SeparableConv, block=ResNeXtBottleneck ):
        super(KHU2, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        depth=38
        self.cardinality=16
        self.base_width=n_feats

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [common.default_conv(args.n_colors, n_feats, kernel_size)]



        # define tail module
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats*4, act=False),
            nn.Conv2d(
                n_feats*4, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.inplanes=n_feats
        self.head = nn.Sequential(*m_head)

        self.tail = nn.Sequential(*m_tail)

        #assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'

        layer_blocks =n_resblock #(depth - 2) // 9



        self.stage_1 = self._make_layer(block, n_feats , layer_blocks, 1)
        #self.stage_2 = self._make_layer(block, 128, layer_blocks, 1)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False))
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))
        return nn.Sequential(*layers)




    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.stage_1(x)
        #x = self.stage_2(x)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
