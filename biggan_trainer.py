import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
import os
import copy
import argparse

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.utils as vutils

from BigGANdeep import Discriminator, Generator
from BigGANdeep2 import Discriminator as D2
from BigGANdeep2 import Generator as G2
from dataset import Cats
from inpaint_tools import read_file_list


class Trainer(object):
    def __init__(self, dataset, params):
        ### Misc ###
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        self.p = params

        ### Make Models ###
        if self.p.biggan2:
            self.netD = D2().to(self.device)
            self.netG = G2(dim_z=self.p.z_size).to(self.device)
            self.y = torch.zeros(self.p.batch_size, device=self.p.device).reshape(self.p.batch_size, 1)
        else:
            self.netD = Discriminator().to(self.device)
            self.netG = Generator(dim_z=self.p.z_size).to(self.device)
        
        if self.p.ngpu > 1:
            self.netD = nn.DataParallel(self.netD)
            self.netG = nn.DataParallel(self.netG)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.p.lrD, betas=(0., 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.p.lrG, betas=(0., 0.999))

        self.scalerD = GradScaler()
        self.scalerG = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.fid = []
        self.fid_epoch = []

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                data = data/255
                data = (data*2)-1
                yield data.permute(0,3,1,2)
        
    def log_train(self, step, fake, real):
        with torch.no_grad():
            self.fid.append(
                FID.fid(fake.float(), real_images=real.float(), device=self.p.device, batch_size=self.p.batch_size)
                )
        d_real, d_fake = self.D_losses[-1]
        print('[%d|%d]\tD(x): %.4f\tD(G(z)): %.4f|%.4f\tFID %.4f'
                    % (step, self.p.niters, d_real, d_fake, self.G_losses[-1], self.fid[-1]), flush=True)

    def log_interpolation(self, step):
        noise = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
    
        with torch.no_grad():
            fake = self.netG(self.fixed_test_noise).detach().cpu()
        torchvision.utils.save_image(
            vutils.make_grid(fake, padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])
            
            
            self.netG.load_state_dict(state_dict['modelG_state_dict'])
            self.netD.load_state_dict(state_dict['modelD_state_dict'])

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])

            self.G_losses = state_dict['lossG']
            self.D_losses = state_dict['lossD']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'modelG_state_dict': self.netG.state_dict(),
        'modelD_state_dict': self.netD.state_dict(),
        'optimizerG_state_dict': self.optimizerG.state_dict(),
        'optimizerD_state_dict': self.optimizerD.state_dict(),
        'lossG': self.G_losses,
        'lossD': self.D_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, step, fake, real):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, fake, real)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step)

    def log_final(self, step, fake, real):
        self.log_train(step, fake, real)
        self.log_interpolation(step)
        self.save_checkpoint(step)

    def D_step(self, step, real):
        for p in self.netD.parameters():
                p.requires_grad = True
        self.netD.zero_grad()

        with autocast():
            noise = torch.randn(real.shape[0], self.p.z_size, dtype=torch.float, device=self.device)

            if self.p.biggan2:
                fake = self.netG(noise, self.y)
            else:
                fake = self.netG(noise)

            if self.p.biggan2:
                errD_real = (nn.ReLU()(1.0 - self.netD(real, self.y))).mean()
                errD_fake = (nn.ReLU()(1.0 + self.netD(fake, self.y))).mean()
            else:
                errD_real = (nn.ReLU()(1.0 - self.netD(real))).mean()
                errD_fake = (nn.ReLU()(1.0 + self.netD(fake))).mean()
                
            errD = errD_fake + errD_real

        self.scalerD.scale(errD).backward()
        self.scalerD.step(self.optimizerD)
        self.scalerD.update()

        for p in self.netD.parameters():
            p.requires_grad = False

        self.D_losses.append((errD_real.item(), errD_fake.item()))

    def G_step(self, step):
        for p in self.netG.parameters():
                p.requires_grad = True

        self.netG.zero_grad()
        with autocast():
            noise = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
            
            if self.p.biggan2:
                fake = self.netG(noise, self.y)
            else:
                fake = self.netG(noise)

            if self.p.biggan2:
                errG = -self.netD(fake, self.y).mean()
            else:
                errG = -self.netD(fake).mean()

        self.scalerG.scale(errG).backward()
        self.scalerG.step(self.optimizerG)
        self.scalerG.update()

        for p in self.netG.parameters():
            p.requires_grad = False
        
        self.G_losses.append(errG.item())

        return fake.detach()

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)

        gen = self.inf_train_gen()
        for p in self.netD.parameters():
                p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False

        print("Starting Training...", flush=True)
        for i in range(step_done, self.p.niters):
            for _ in range(self.p.iterD):    
                data = next(gen)
                real = data.to(self.device)
                self.D_step(i, real)

            fake = self.G_step(i)

            self.log(i, fake, real)
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
        #self.tracker.stop()
        self.log_final(i, fake, real)
        print('...Done', flush=True)


def main():
    parser = argparse.ArgumentParser()
    ## MISC & Hyper
    parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--z_size', type=int, default=512, help='Latent space dimension')
    parser.add_argument('--iterD', type=int, default=2, help='Number of D iters per iter')
    parser.add_argument('--lrG', type=float, default=5e-5, help='Learning rate G')
    parser.add_argument('--lrD', type=float, default=5e-5, help='Learning rate D')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
    parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
    parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
    parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
    parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
    parser.add_argument('--biggan2', type=bool, default=False)
    args = parser.parse_args()


    file_list = os.path.join("./MissingDataOpenData/", "data_splits", "training.txt")
    file_ids = read_file_list(file_list)
    dataset_train = Cats(path="./MissingDataOpenData/", files_orig=file_ids)

    trainer = Trainer(dataset_train, params=args)
    trainer.train()

if __name__ == '__main__':
    main()



