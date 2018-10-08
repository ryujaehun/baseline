import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import subprocess
import model_estimate
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
path=''
if args.recursive is True:
    model_estimate.main(args)
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

path='/'+'/'.join(os.getcwd().split('/')[1:-1])
path=os.path.join(path,'experiment')
path=os.path.join(path,args.save)
path+='/model/model_best_x'+str(args.scale[0])+'.pth.tar'
if args.recursive is True:
    for i in ['Set5','Set14','B100','Urban100','DIV2K']:
        if path !='':
            proc = subprocess.Popen( 'python3 main.py  --save '+str(args.save)+'/'+i+' --model '+args.model+' --data_test '+i+' --ext img --scale '+str(args.scale[0])+' --n_resblocks '+str(args.n_resblocks)+' --n_feats '+str(args.n_feats)+' --res_scale '+str(args.res_scale)+' --pre_train '+path+' --test_only --save_results --recursive False', shell=True, executable='/bin/bash')
            proc.communicate()
