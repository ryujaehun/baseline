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
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
path=''
model_estimate.main(args)
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
        path='/'+'/'.join(os.getcwd().split('/')[1:-1])
        path=os.path.join(path,'experiment')
        path=os.path.join(path,args.save)
        path+='/model/model_best.pth.tar'

    checkpoint.done()
for i in ['B100','Set14','Set5','Urban100','DIV2K']:
    if path !='':
        proc = subprocess.Popen( 'python3 main.py  --save '+str(args.save)+'/'+i+' --data_test '+i+' --ext img --scale '+str(args.scale[0])+' --n_resblocks '+str(args.n_resblocks)+' --n_feats '+str(args.n_feats)+' --res_scale '+str(args.res_scale)+' --pre_train '+path+' --test_only --save_results', shell=True, executable='/bin/bash')
        proc.communicate()
