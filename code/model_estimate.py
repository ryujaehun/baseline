import option
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
resolution={'HD':(1280,720),'FHD':(1920,1080),'QHD':(2560,1440),'UHD':(3840,2160)}

list_model=['srcnn','fsrcnn','vdsr','kaist','edsr']#,'khu1','khu2','khu3','khu4']


def print_model_parm_nums(my_models):
    model = my_models
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in filter(lambda x: x.requires_grad, model.parameters())])
    temp=[]
    temp.append(str(int(num_params) / 1e6)[:-2]+' M')
    temp.append(str(int(num_params) / 1e6*4)[:-2]+' M')
    return temp

def print_model_parm_flops(my_models,_input=torch.rand(1,3,224,224)):
    multiply_adds = False
    list_conv=[]
    list_conv_feature=[]
    def conv_hook(self, input, output):
        Tt=lambda x:torch.tensor(x)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = Tt(self.kernel_size[0]) * Tt(self.kernel_size[1]) * Tt(int(self.in_channels / self.groups))* Tt(2 if multiply_adds else 1)
        bias_ops = Tt(1 if self.bias is not None else 0)
        params = Tt(output_channels) * Tt(kernel_ops + bias_ops)
        flops = Tt(batch_size) * Tt(params) * Tt(output_height) * Tt(output_width)
        list_conv.append(flops)

        if self.groups == 1:

            list_conv_feature.append(Tt(batch_size) * Tt(output_channels) * Tt(output_height) * Tt(output_width))
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
    list_pooling_feature=[]
    def pooling_hook(self, input, output):
        Tt=lambda x:torch.tensor(x)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops =Tt(self.kernel_size) * Tt(self.kernel_size)
        bias_ops = 0
        params = Tt(output_channels) * Tt(kernel_ops + bias_ops)
        flops = Tt(batch_size) * Tt(params) * Tt(output_height) * Tt(output_width)
        list_pooling.append(flops)
        list_pooling_feature.append(Tt(batch_size) * Tt(output_channels) * Tt(output_height) * Tt(output_width))

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

    model =my_models.cuda()
    register_hook(model)
    with torch.no_grad():
        input = _input.cuda()
        out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total_feature = sum(list_conv_feature) #+ sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling_feature))
    temp=[]
    temp.append(str(total_flops.item() / 1e9)[:-2]+' G')
    temp.append(str(total_feature.item()*4/1e6)[:-2]+' M')
    temp.append(int(len(list_conv)))
    return temp



def main(args):
    size=resolution[args.resolution]
    if not args.model in list_model:
        list_model.append(args.model)
    path='/'+'/'.join(os.getcwd().split('/')[1:-1])
    path=os.path.join(path,'experiment')
    path=os.path.join(path,args.save)
    result=dict()

    for models in list_model:
        if models.upper() == 'EDSR':
            from model.edsr import EDSR
            _model=EDSR(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)
        elif models.upper() == 'EDSR_MOBILE':
            from model.edsr_mobile import EDSR
            _model=EDSR(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)
        elif models.upper() == 'VDSR':
            from model.vdsr import VDSR
            _model=VDSR()
            input=torch.rand(1,1,size[0],size[1])
        elif models.upper() == 'SRCNN':
            from model.srcnn import SRCNN
            _model=SRCNN()
            input=torch.rand(1,1,size[0],size[1])
        elif models.upper() == 'ESPCN':
            from model.espcn import ESPCN
            _model=ESPCN()
            input=torch.rand(1,1,size[0]//2,size[1]//2)
        elif models.upper() == 'KAIST':
            from model.kaist import KAIST
            _model=KAIST()
            input=torch.rand(1,1,size[0]//2,size[1]//2)
        elif models.upper() == 'MDSR':
            from model.mdsr import MDSR
            _model=MDSR(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)

        elif models.upper() == 'KHU1':
            from model.khu1 import KHU1
            _model=KHU1(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)

        elif models.upper() == 'KHU2':
            from model.khu2 import KHU2
            _model=KHU2(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)

        elif models.upper() == 'KHU3':
            from model.khu3 import KHU3
            _model=KHU3(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)
        elif models.upper() == 'KHU4':
            from model.khu4 import KHU4
            _model=KHU4(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)

        elif models.upper() == 'MDSR_MOBILE':
            from model.mdsr_mobile import MDSR
            _model=MDSR(args)
            input=torch.rand(1,3,size[0]//2,size[1]//2)
        elif models.upper() == 'FSRCNN':
            from model.fsrcnn import FSRCNN
            _model=FSRCNN( num_channels=1, upscale_factor=args.scale)
            input=torch.rand(1,1,size[0]//2,size[1]//2)
        elif models.upper() == 'DRCN':
            from model.drcn import DRCN
            _model=DRCN(num_channels=1,base_channel=256,num_recursions=16)
            input=torch.rand(1,1,size[0],size[1])
        else:
            raise NotImplementedError("To be implemented")

        temp=print_model_parm_flops(_model,input)
        temp.extend(print_model_parm_nums(_model))
        result[models]=temp

    df=pd.DataFrame(result,index=['# of Gflops (FHD)','feature size','# of layer',"# of Params",'model size'])
    df=df.reindex_axis(list_model, axis=1)
    df=df.reindex(["# of Params",'# of Gflops (FHD)','# of layer','feature size','model size'])
    df.to_html(path+'/report_x'+str(args.scale[0])+'.html')
    df.to_csv(path+'/report_x'+str(args.scale[0])+'.csv')
    df.to_latex(path+'/report_x'+str(args.scale[0])+'.latex')



if __name__ == '__main__':
    main(option.args)
