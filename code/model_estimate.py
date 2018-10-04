from option import args
import torch
import torchvision
import torch.nn as nn
from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import pandas as pd
from model.vdsr import *
from model.srcnn import *
from model.espcn import *
from model.kaist import *

from pandas import DataFrame
from graphviz import Digraph

index=['total flops','num of parameters','estimated of size']
total=DataFrame(data=np.zeros((1,3)),columns=index)
total.name=args.model
def print_model_parm_nums(my_models=VDSR()):
    model = my_models#EDSR(args)
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in filter(lambda x: x.requires_grad, model.parameters())])
    total['num of parameters']=(int(num_params) )
    total['estimated of size']=(int(num_params) *4)
    print('num of parameters: %.4fM ' % (int(num_params) / 1e6) +'\n')
    print('estimated of size %.4fM ' % (int(num_params) / 1e6*4) +'\n')
def print_model_parm_flops(my_models=models.resnet50(),_input=torch.rand(1,3,224,224)):
    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        Tt=lambda x:torch.tensor(x)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = Tt(self.kernel_size[0]) * Tt(self.kernel_size[1]) * Tt(int(self.in_channels / self.groups))* Tt(2 if multiply_adds else 1)
        bias_ops = Tt(1 if self.bias is not None else 0)
        params = Tt(output_channels) * Tt(kernel_ops + bias_ops)
        flops = Tt(batch_size) * Tt(params) * Tt(output_height) * Tt(output_width)
        list_conv.append(flops)
    list_linear=[]
    def linear_hook(self, input, output):

        Tt=lambda x:torch.tensor(x)
        batch_size = Tt(input[0].size(0) if input[0].dim() == 2 else 1)

        weight_ops = Tt(self.weight.nelement()) * Tt(2 if multiply_adds else 1)
        bias_ops = Tt(self.bias.nelement())
        flops = Tt(batch_size) * Tt(weight_ops + bias_ops)
        list_linear.append(flops)
    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())


    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        Tt=lambda x:torch.tensor(x)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops =Tt(self.kernel_size) * Tt(self.kernel_size)
        bias_ops = 0
        params = Tt(output_channels) * Tt(kernel_ops + bias_ops)
        flops = Tt(batch_size) * Tt(params) * Tt(output_height) * Tt(output_width)
        list_pooling.append(flops)

    def register_hook(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
                register_hook(c)

    model =my_models
    register_hook(model)
    with torch.no_grad():
        input = _input
        out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total['total flops']=total_flops.item()
    print('Total Number of FLOPs: %.3fG' % (total_flops.item() / 1e9)+'\n')
    _index=['conv','linear','batch norm','activation','pooling']
    layer_csv=DataFrame(data=np.zeros((max([len(list_conv),len(list_linear),len(list_relu),len(list_pooling)]),5)),columns=_index)

    for i in range(len(list_conv)):
        layer_csv['conv'][i]=list_conv[i].item()
        print(' Conv  layer `s '+str(i+1) + 'f FLOPs: %.3fG' % (list_conv[i].item() / 1e9)+'\n')
    for i in range(len(list_linear)):
        print(' linear layer `s '+str(i+1) +' of FLOPs: %.3fG' % (list_linear[i].item() / 1e9)+'\n')
        layer_csv['linear'][i]=list_linear[i].item()
    for i in range(len(list_bn)):
        print(' batch norm layer `s '+str(i+1) +' of FLOPs: %.3fG' % (list_bn[i].item() / 1e9)+'\n')
        layer_csv['batch norm'][i]=list_bn[i].item()
    for i in range(len(list_relu)):
        print(' activation layer `s '+str(i+1) +' of FLOPs: %.4fG' % (list_relu[i] / 1e9)+'\n')
        layer_csv['activation'][i]=list_relu[i]
    for i in range(len(list_pooling)):
        print(' pooling layer `s '+str(i+1) +' of FLOPs: %.3fG' % (list_pooling[i].item() / 1e9)+'\n')
        layer_csv['pooling'][i]=list_pooling[i].item()
    layer_csv.to_csv(args.model+'_layer.csv',index=False)

def visual(_model=models.AlexNet(),_input=torch.randn(1, 3, 224, 224).requires_grad_(True),filename=args.model):
    model =_model#VDSR()

    x = _input
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)])).render(filename=filename)

def main():

    if args.model == 'EDSR':
        from model.edsr import EDSR
        _model=EDSR(args)
        input=torch.rand(1,3,224,224)
    elif args.model == 'EDSR_MOBILE':
        from model.edsr_mobile import EDSR
        _model=EDSR(args)
        input=torch.rand(1,3,224,224)
    elif args.model == 'VDSR':
        _model=VDSR()
        input=torch.rand(1,1,224,224)
    elif args.model == 'SRCNN':
        _model=SRCNN()
        input=torch.rand(1,1,224,224)
    elif args.model == 'ESPCN':
        _model=ESPCN()
        input=torch.rand(1,1,224,224)
    elif args.model == 'KAIST':
        _model=KAIST()
        input=torch.rand(1,1,224,224)
    elif args.model == 'MDSR':
        from model.mdsr import MDSR
        _model=MDSR(args)
        input=torch.rand(1,3,224,224)
    elif args.model == 'MDSR_MOBILE':
        from model.mdsr_mobile import MDSR
        _model=MDSR(args)
        input=torch.rand(1,3,224,224)

    else:
        raise NotImplementedError("To be implemented")


    print_model_parm_flops(_model,input)
    print_model_parm_nums(_model)
    #visual(_model,input)
    total.to_csv(args.model+'_total.csv',index=False)

if __name__ == '__main__':
    main()
