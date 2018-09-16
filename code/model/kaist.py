import torch
import torch.nn as nn
from math import sqrt

#  this implementation  only use estimated model!
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        #(5,1),(1,1),(2,0),groups=32,bias=True
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1), stride=(1,1), padding=(2,0), groups=32,bias=False)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,1), stride=(1,1), padding=(2,0),groups=16, bias=False)
        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        residual=x
        out=self.conv_2(self.conv_1(self.relu(x)))
        out=self.conv_4(self.conv_3(self.relu(out)))
        out = torch.add(out,residual)
        return out

class KAIST(nn.Module):
    def __init__(self):
        super(KAIST, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 1)
        self.input = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1,1), padding=1,groups=32, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1,groups=16, bias=False)
        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.input(x)
        out = self.residual_layer(out)
        out = self.relu(out)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.relu(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = torch.add(out,residual)
        return out
