import os
import math
from decimal import Decimal
import datetime
import utility
import torch
from torch.autograd import Variable
from tqdm import tqdm
import ssim
import pandas as pd


list_testset=['Set5','Set14','B100','Urban100','DIV2K']

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_x'+str(self.args.scale[0])+'.pth.tar'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        acc=0
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                t1=timer_model.release()
                t2=timer_data.release()
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    t1,
                    t2))

                acc+=(t1+t2)

            timer_data.tic()
        print('traing finished ',str(datetime.timedelta(seconds=round(acc*self.args.epochs-epoch))),'left. {} epoch left.'.format(self.args.epochs-epoch))
        acc=0
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                if self.args.test_only:
                    eval_acc_ssim=0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):

                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)



                    save_list = [sr]
                    if not no_eval:

                        if self.args.test_only:
                            eval_acc_ssim+=ssim.ssim(sr, hr).item()


                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)


                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)

                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        if self.args.test_only:
            path='/'+'/'.join(os.getcwd().split('/')[1:-1])
            path=os.path.join(path,'experiment')
            path=os.path.join(path,self.args.save.split('/')[0])


            if os.path.exists(path+'/result.h5'):
                df=pd.read_hdf(path+'/result.h5',key='results')
            else:
                cols = pd.MultiIndex.from_tuples([("Set5","PSNR"),("Set5","SSIM"),("Set14","PSNR"),("Set14","SSIM")
                                 ,("B100","PSNR"),("B100","SSIM"),("Urban100","PSNR"),("Urban100","SSIM"),("DIV2K","PSNR"),("DIV2K","SSIM")])
                df=pd.DataFrame(columns=cols,index=[2,3,4])
            _psnr=eval_acc / len(self.loader_test)
            _ssim=eval_acc_ssim / len(self.loader_test)
            temp=df[self.args.data_test]
            temp['PSNR'][self.args.scale]=_psnr
            temp['SSIM'][self.args.scale]=_ssim
            df[self.args.data_test]=temp
            df.to_hdf(path+'/result.h5',key='results')
            print('PSNR: ',_psnr)
            print('SSIM: ',_ssim)
            if self.args.data_test =='DIV2K':
                df.to_html(path+'/result.html')

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
