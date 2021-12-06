from __future__ import print_function
import torch, os, argparse, torchvision, torch.utils.data, time
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
from WGAN import WGAN
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pytorch_fid import fid_score
#TODO Add argparsing
#TODO add inception calculations stuff
torch.backends.cudnn.benchmark = True

image_iteration = 100 #after how many batches should an image samples be generated

amp = True
num_gpu = 1
num_workers = 4
image_size = 128
num_features = image_size
n_convolution_blocks = 4
batch_size = 64
latent_vector_size =128
num_epochs = 30
model_load_path = None
model_save_folder = 'checkpoints-WGAN'
IMAGE_PATH ='/home/ej74/Resized' #'/input/flickrfaceshq-dataset-nvidia-resized-256px'
IMAGE_PATH2 ='/home/ej74/CelebA/img_align_celeba'#'celeba-dataset/img_align_celeba/'
epoch = 0

precomputed_inception_score_path = ('home/ej74/128px.nz') #m1, s1




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


GAN=WGAN(num_gpu, num_features, n_convolution_blocks,latent_vector_size=latent_vector_size,AMP=amp)

device = GAN.device
dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([dataset,dataset2]), batch_size=batch_size, shuffle=True, num_workers=num_workers)

if model_load_path is not None:
    total_epoch = GAN.load_checkpoint(model_load_path)
    #override amp from model load
    if amp:
        GAN.amp = True

else:
    total_epoch = 0 


writer = SummaryWriter()

fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device) #fixed noise for plotting
print("Starting Training Loop...")
for epoch in range(num_epochs):
    #Per batch
    start_time = time.time()
    errD = 0
    errG = 0
    for i, data in enumerate(dataloader, 0):
        images = data[0]
        loss_d, D_x, D_g_z= GAN.train_discriminator(images)
        loss_g = GAN.train_generator(batch_size)
	
        writer.add_scalar('Generator Loss',loss_g,GAN.iters)
        writer.add_scalar('Discriminator Loss',loss_d,GAN.iters)
        writer.add_scalar('Critic Real Score',D_x,GAN.iters)
        writer.add_scalar('Critic Fake Score',D_g_z,GAN.iters)

        if i % 50 == 0:

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tCritic Real: %.4f\tCritic Fake %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss_d, loss_g, D_x, D_g_z))
           
        if GAN.iters % image_iteration == 0 or (epoch == num_epochs-1 and i == len(dataloader)-1):
            with torch.no_grad():
                fake_images=GAN.generator(fixed_noise).detach().cpu()
            img_grid=vutils.make_grid(fake_images, padding=2, normalize=True)
            writer.add_image('Fake',img_grid,GAN.iters)

            img_grid=vutils.make_grid(data[0].to(device)[:64], padding=2, normalize=True).cpu()
            writer.add_image('Real Images', img_grid)
    
    print((time.time()-start_time)//60,'minutes elapsed this epoch')

    
    GAN.save_checkpoint(model_save_folder,total_epoch+epoch,f'epoch{total_epoch+epoch}_model.pt')
    




