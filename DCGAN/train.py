from __future__ import print_function
import torch, os, argparse, torchvision, torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import DCGAN
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
#TODO Add argparsing
#TODO add inception calculations stuff
torch.backends.cudnn.benchmark = True


num_gpu = 1
num_workers = 4
image_size = 128
num_features = image_size
n_convolution_blocks = 4
batch_size = 128
latent_vector_size =128
num_epochs = 1


IMAGE_PATH ='/home/ej74/Resized' #'/input/flickrfaceshq-dataset-nvidia-resized-256px'
IMAGE_PATH2 ='/home/ej74/CelebA/img_align_celeba'#'celeba-dataset/img_align_celeba/'

epoch = 0


dataset = dset.ImageFolder(root=IMAGE_PATH,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset2 = dset.ImageFolder(root=IMAGE_PATH2,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


GAN=DCGAN(num_gpu, num_features, n_convolution_blocks,latent_vector_size=latent_vector_size)
print(GAN.generator)
print(summary(GAN.generator, (latent_vector_size,1,1)))
print(summary(GAN.discriminator,(3,64,64) ))
device = GAN.device
dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([dataset,dataset2]), batch_size=batch_size, shuffle=True, num_workers=num_workers)

writer = SummaryWriter()

fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device) #fixed noise for plotting
print("Starting Training Loop...")
for epoch in range(num_epochs):
    #Per batch
    for i, data in enumerate(dataloader, 0):
        images = data[0]
        errD = GAN.train_discriminator(images)
        errG = GAN.train_generator(batch_size)

        writer.add_scalar('Generator Loss',errG,GAN.iters)
        writer.add_scalar('Discriminator Loss',errD,GAN.iters)

        if i % 50 == 0: 
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD, errG))

        if GAN.iters % 500 == 0 or (epoch == num_epochs-1 and i == len(dataloader)-1):
            with torch.no_grad():
                fake_images=GAN.generator(fixed_noise).detach().cpu()
            img_grid=vutils.make_grid(fake_images, padding=2, normalize=True)
            writer.add_image('Fake',img_grid,GAN.iters)

            img_grid=vutils.make_grid(data[0].to(device)[:64], padding=2, normalize=True).cpu()
            writer.add_image('Real Images', img_grid)

    GAN.save_checkpoint('checkpoints',epoch,f'epoch{epoch}_model.pt')





