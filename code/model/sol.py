from model import common
import torch.nn.init as init
import torch.nn as nn

def make_model(args, parent=False):
    return SOL(args)

class SOL(nn.Module):
    def __init__(self, args):
        super(SOL, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        self.conv6 = nn.Conv2d(32, args.scale[0] ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(args.scale[0])
        self._initialize_weights()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual=x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x+=residual
        x = self.pixel_shuffle(self.conv6(x))
        return x
    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv6.weight)


    def load_state_dict(self, state_dict, strict=True):
        pass
