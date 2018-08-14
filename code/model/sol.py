from model import common
import torch.nn.init as init
import torch.nn as nn

def make_model(args, parent=False):
    return SOL(args)

class SOL(nn.Module):
    def __init__(self, args):
        super(SOL, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1))
        self.convdepthwise2=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=(1,1),bias=False)
        self.convpointwise2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0,bias=False)
        self.convdepthwise3=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=(1,1),bias=False)
        self.convpointwise3=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0,bias=False)
        self.convdepthwise4=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=(1,1),bias=False)
        self.convpointwise4=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0,bias=False)
        self.convdepthwise5=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.convpointwise5=nn.Conv2d(in_channels=32,out_channels=args.n_colors*args.scale[0] ** 2,kernel_size=1,stride=1,padding=0,bias=False)
        self.pixel_shuffle = nn.PixelShuffle(args.scale[0])
        self._initialize_weights()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual=x
        x = self.relu(self.convpointwise2(self.convdepthwise2(x)))
        x = self.relu(self.convpointwise3(self.convdepthwise3(x)))
        x = self.relu(self.convpointwise3(self.convdepthwise3(x)))
        x = self.relu(self.convpointwise4(self.convdepthwise4(x)))
        x+=residual
        x=self.relu(self.convpointwise5(self.convdepthwise5(x)))
        x = self.pixel_shuffle(x)
        return x
    def _initialize_weights(self):
        init.kaiming_normal_(self.convdepthwise2.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convpointwise2.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convdepthwise3.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convpointwise3.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convdepthwise4.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convpointwise4.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convdepthwise5.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.convpointwise5.weight, nonlinearity='leaky_relu')
