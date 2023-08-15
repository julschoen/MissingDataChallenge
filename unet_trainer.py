import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
import os
import copy
import argparse
from pytorch_msssim import ssim

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.utils as vutils
import torch.nn.functional as F

from unet_model import UNet
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
        self.unet = UNet().to(self.p.device)

        self.opt = optim.Adam(self.unet.parameters(), lr=self.p.lr, betas=(0., 0.9))

        self.scaler = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.losses = []

    def inf_train_gen(self):
        while True:
            for x,y in self.generator_train:
                yield x.permute(0,3,1,2), y.permute(0,3,1,2)
        
    def log_train(self, step):
        l1, l2 = self.losses[-1]
        print('[%d|%d]\tMSE: %.4f\tSSIM: %.4f' % (step, self.p.niters, l1, l2), flush=True)

    def log_interpolation(self, step, fake, real):
        torchvision.utils.save_image(
            vutils.make_grid(fake, padding=2, normalize=True)
            , os.path.join(self.images_dir, f'rec_{step}.png'))

        torchvision.utils.save_image(
            vutils.make_grid(real, padding=2, normalize=True)
            , os.path.join(self.images_dir, f'target_{step}.png'))


    def start_from_checkpoint(self):
        step = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']

            self.opt.load_state_dict(state_dict['opt'])
            
            self.unet.load_state_dict(state_dict['model'])

            self.losses = state_dict['loss']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'model': self.unet.state_dict(),
        'opt': self.opt.state_dict(),
        'loss': self.losses
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, step, fake, real):
        if step % self.p.steps_per_log == 0:
            self.log_train(step)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step, fake, real)

    def log_final(self, step, fake, real):
        self.log_train(step)
        self.log_interpolation(step, fake, real)
        self.save_checkpoint(step)


    def train(self):
        step_done = self.start_from_checkpoint()

        gen = self.inf_train_gen()
        for p in self.unet.parameters():
                p.requires_grad = True

        print("Starting Training...", flush=True)
        for i in range(step_done, self.p.niters):
            im, masked = next(gen)
            im = im.to(self.p.device)
            masked = masked.to(self.p.device)

            rec = self.unet(masked)
            mse_loss = F.mse_loss(rec, im)
            ssim_loss = 1 - ssim(rec, im, data_range=255)

            if self.p.only_ssim:
                loss = ssim_loss
            else:
                loss = mse_loss + ssim_loss

            loss.backward()
            self.opt.step()

            self.losses.append((mse_loss.detach().item(), ssim_loss.detach().item()))
            self.log(i, rec, im)
            if i%100 == 0 and i>0:
                self.save_checkpoint(i)
        
        for p in self.unet.parameters():
                p.requires_grad = False
        self.log_final(i, rec, im)
        print('...Done', flush=True)


def main():
    parser = argparse.ArgumentParser()
    ## MISC & Hyper
    parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate D')
    parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
    parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
    parser.add_argument('--log_dir', type=str, default='unet', help='Save Location')
    parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
    parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
    parser.add_argument('--only_ssim', type=bool ,default=False)
    args = parser.parse_args()


    file_list = os.path.join("./MissingDataOpenData/", "data_splits", "training.txt")
    file_ids = read_file_list(file_list)
    dataset_train = Cats(path="./MissingDataOpenData/", files_orig=file_ids, files_masked=file_ids)

    trainer = Trainer(dataset_train, params=args)
    trainer.train()

if __name__ == '__main__':
    main()



