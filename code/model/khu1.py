from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return KHU1(args, dilated.dilated_conv)
    else:
        return KHU1(args)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0,act=nn.ReLU(inplace=True)):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.relu = act
        self.conv1 = common.default_conv(in_planes, inter_planes, kernel_size=1,
                              bias=False)
        self.conv2 = common.default_conv(inter_planes, out_planes, kernel_size=3,
                                bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1((x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0,act=nn.ReLU(inplace=True)):
        super(TransitionBlock, self).__init__()

        self.relu = act
        self.conv1 = common.default_conv(in_planes, out_planes, kernel_size=1,
                               bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0,act=nn.ReLU(inplace=True)):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate,act)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate,act):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate,act))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)



class KHU1(nn.Module):
    def __init__(self, args,growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(KHU1, self).__init__()
        depth=args.n_resblocks

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        #in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        in_planes=n_feats
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate,act=act)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,act=act)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate,act=act)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,act=act)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate,act=act)
        in_planes = int(in_planes+n*growth_rate)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        m_head = [common.default_conv(args.n_colors, n_feats, kernel_size)]

        m_tail = [
            common.Upsampler(common.default_conv, scale, in_planes, act=False),
            nn.Conv2d(
                in_planes, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x=self.tail(x)
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
