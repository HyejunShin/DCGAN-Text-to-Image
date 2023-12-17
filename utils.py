import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from torch import nn
from torch import  autograd
import torch
import torchvision.utils as vutils
import pdb
import json

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class minibatch_discriminator(nn.Module):
    def __init__(self, num_channels, B_dim, C_dim):
        super(minibatch_discriminator, self).__init__()
        self.B_dim = B_dim
        self.C_dim =C_dim
        self.num_channels = num_channels
        T_init = torch.randn(num_channels * 4 * 4, B_dim * C_dim) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)

    def forward(self, inp):
        inp = inp.view(-1, self.num_channels * 4 * 4)
        M = inp.mm(self.T_tensor)
        M = M.view(-1, self.B_dim, self.C_dim)

        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)

        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1)

        output = torch.cat((inp, output), 1)

        return output


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod

    # based on:  https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def compute_GP(netD, real_data, real_embed, fake_data, LAMBDA):
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates, real_embed)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Logger(object):
    def __init__(self, iter_log_file, epoch_log_file):
        self.iter_log_file = iter_log_file
        self.epoch_log_file = epoch_log_file
        # Initialize the arrays to store metrics
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []
        self.image_counter = 0

    def log_iteration_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss):
        message = f"Epoch: {epoch}, Gen_iteration: {gen_iteration}, d_loss= {d_loss.data.cpu().mean()}, g_loss= {g_loss.data.cpu().mean()}, real_loss= {real_loss}, fake_loss = {fake_loss}"
        print(message)
        self.log_to_file(self.iter_log_file, message)
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())

    def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
        message = f"Epoch: {epoch}, d_loss= {d_loss.data.cpu().mean()}, g_loss= {g_loss.data.cpu().mean()}, D(X)= {real_score.data.cpu().mean()}, D(G(X))= {fake_score.data.cpu().mean()}"
        print(message)
        self.log_to_file(self.iter_log_file, message)
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    def log_epoch(self, epoch):
        # Log the average values
        avg_d_loss = float(np.array(self.hist_D).mean())
        avg_g_loss = float(np.array(self.hist_G).mean())
        log_data = {
            "epoch": epoch,
            "avg_d_loss": avg_d_loss,
            "avg_g_loss": avg_g_loss
        }
        self.log_to_file(self.epoch_log_file, json.dumps(log_data))
        self.hist_D = []
        self.hist_G = []

    def log_epoch_w_scores(self, epoch):
        # Log the average values along with scores
        avg_d_loss = float(np.array(self.hist_D).mean())
        avg_g_loss = float(np.array(self.hist_G).mean())
        avg_dx = float(np.array(self.hist_Dx).mean())
        avg_dgx = float(np.array(self.hist_DGx).mean())
        log_data = {
            "epoch": epoch,
            "avg_d_loss": avg_d_loss,
            "avg_g_loss": avg_g_loss,
            "avg_dx": avg_dx,
            "avg_dgx": avg_dgx
        }
        self.log_to_file(self.epoch_log_file, json.dumps(log_data))
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def save_img(self, right_images, fake_images, epoch):
        # Ensure the directory for the epoch exists
        image_dir = './images'
        epoch_dir = os.path.join(image_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        self.image_counter += 1

        # Save real images
        real_image_file = os.path.join(epoch_dir, f'real_images_batch_{self.image_counter}.png')
        vutils.save_image(right_images, real_image_file, normalize=True)

        # Save fake images
        fake_image_file = os.path.join(epoch_dir, f'fake_images_batch_{self.image_counter}.png')
        vutils.save_image(fake_images, fake_image_file, normalize=True)

    def log_to_file(self, log_file, message):
        # Open the specified log file and append the message
        with open(log_file, 'a') as logf:
            logf.write(message + '\n')
