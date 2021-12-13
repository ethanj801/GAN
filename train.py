##add pretraining of critic (modifiable) for WGAN
from __future__ import print_function
import argparse
import math
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
from DCGAN import DCGAN
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pytorch_fid import fid_score
#TODO add inception calculations stuff

parser = argparse.ArgumentParser()
parser.add_argument('amp',type =bool, default = True, help='Whether to use Automatic Mixed Precision')
parser.add_argument('generator_training_interval',type =int, default = 1, help='How often to train generator with respect to critic. e.g. a value of 3 means generator is trained every 3 critic updates')
parser.add_argument('gpu',type =int, default = 1, help='Number of GPUs to use')
parser.add_argument('workers',type =int, default = 4, help='Number of CPU threads to use to load data')
parser.add_argument('image_size',type =int, default = 128, help='Size of images')
parser.add_argument('batch_size',type =int, default = 64, help='Batch size for training')
parser.add_argument('model_load_path',type =str, default = None, help='Path to pretrained model')
parser.add_argument('model_save_path',type =str, default = 'checkpoints', help='Path to model save folder')
parser.add_argument('epochs',type =int, default = 30, help='Number of epochs to train for')
parser.add_argument('checkpoint_save_frequency',type =int, default = 3, help='After how many epochs to save model')
parser.add_argument('image_iteration',type =int,default=500,help='After how many batches should an image samples be generated')
parser.add_argument('precomputed_inception_score_path',type =str,default="home/ej74/128px.nz",help='Path to precomputed inception scores')
parser.add_argument('nz',type =int,default=256,help='Latent vector size')
parser.add_argument('model_type',type =str,default='DCGAN',help='Model to use')
args=parser.parse_args()

model_choices = {'DCGAN':DCGAN,'WGAN':WGAN}
if parser.model_type is not in model_choices:
    raise Exception('Invalid Model Type')

torch.backends.cudnn.benchmark = True
scratch_directory = '/n/scratch3/users/e/ej74'
image_iteration = parser.image_iteration 
generator_training_interval = args.generator_training_interval
amp = args.amp
num_gpu = parser.gpu
num_workers = parser.workers
image_size = parser.image_size
num_features = image_size
n_convolution_blocks = math.log2(image_size)-2
batch_size = parser.batch_size
latent_vector_size =parser.nz
num_epochs = parser.epochs
model_load_path = parser.model_load_path
model_save_folder = parser.model_save_path
checkpoint_save_frequency = parser.checkpoint_save_frequency

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
                               transforms.RandomHorizontalFlip()]))

dataset2 = dset.ImageFolder(root=IMAGE_PATH2,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.RandomHorizontalFlip()]))


GAN=model_choices[parser.model_type](num_gpu, num_features, n_convolution_blocks,latent_vector_size=latent_vector_size,AMP=amp)

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
    
    for i, data in enumerate(dataloader, 0):
        images = data[0]
        loss_d, critic_score_real, critic_score_fake = GAN.train_discriminator(images)
        if i%generator_training_interval==0:
            loss_g = GAN.train_generator(batch_size)
            writer.add_scalar('Generator Loss',loss_g,GAN.iters)

        writer.add_scalar('Discriminator Loss',loss_d,GAN.iters)
        writer.add_scalar('Critic Real Score',critic_score_real,GAN.iters)
        writer.add_scalar('Critic Fake Score',critic_score_fake,GAN.iters)

        if i % 50 == 0:

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tCritic Real: %.4f\tCritic Fake %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss_d, loss_g, critic_score_real, critic_score_fake))
           
        if GAN.iters % image_iteration == 0 or (i == len(dataloader)-1):
            with torch.no_grad():
                fake_images=GAN.generator(fixed_noise).detach().cpu()
            img_grid=vutils.make_grid(fake_images, padding=2, normalize=True)
            writer.add_image('Fake',img_grid,GAN.iters)

            img_grid=vutils.make_grid(data[0].to(device)[:64], padding=2, normalize=True).cpu()
            writer.add_image('Real Images', img_grid)
    
    print((time.time()-start_time)//60,'minutes elapsed this epoch')

    if epoch %checkpoint_save_frequency==0 or epoch ==num_epochs-1:
        GAN.save_checkpoint(model_save_folder,total_epoch+epoch,f'epoch{total_epoch+epoch}_model.pt')
    




