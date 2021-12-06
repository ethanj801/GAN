import torch, os, torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import Discriminator, Generator
from utilities import weights_init, requires_grad
from torch.nn import functional as F
class WGAN():
    """Complete WGAN model.
    
    Attributes: #TODO

    """

    def __init__(self,num_gpu, num_features, n_convolution_blocks,lr = 0.00005,
                 latent_vector_size=128, AMP = False):
        """Constructor for DCGAN Class.
        
        Args:
            num_gpu: Number of GPUs to use.
            num_features: Size of the image.
            n_convolution_blocks: Number of convolutional block in Generator and Discriminator.
            lr: Learning rate for optimizers.
            latent_vector_size: Size of latent vector for Generator
            AMP: When True, enables Automatic Mixed Precision.


        """
        self.num_gpu = num_gpu
        self.num_features = num_features
        self.latent_vector_size=latent_vector_size
        self.amp = AMP
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_gpu > 0) else "cpu")
        self.iters = 0 #number of discriminator training iterations

        #Set up models. 
        if self.amp:
            self.scalerD = torch.cuda.amp.GradScaler()
            self.scalerG = torch.cuda.amp.GradScaler()

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_blocks,sigmoid=False).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = latent_vector_size).to(self.device)

        else:
            self.scalerD = None
            self.scalerG = None

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_blocks,sigmoid=False).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = latent_vector_size).to(self.device)


        #Multi GPU compatibility -- untested as of Dec 4 2021
        if self.device.type == 'cuda' and self.num_gpu > 1:
            self.discriminator = nn.DataParallel(self.discriminator, list(range(self.num_gpu)))
            self.generator = nn.DataParallel(self.generator, list(range(self.num_gpu)))
        
        #Initialize weights as per DCGAN paper
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        #Use RMS prop optimizers as outlined in paper
        self.optimizerD = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.optimizerG = optim.RMSprop(self.generator.parameters(), lr=lr)
    

    def save_checkpoint(self,directory,epoch = None,model_name = 'model.pt'):
        """Saves a model checkpoint."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        out = {'epoch': epoch,
            'model_state_dict_G': self.generator.state_dict(),
            'optimizer_state_dict_G': self.optimizerG.state_dict(),
            'model_state_dict_D': self.discriminator.state_dict(),
            'optimizer_state_dict_D': self.optimizerD.state_dict(),
            'iters':self.iters,
            'amp': False #changed below if amp is enabled
            }
        if self.amp:
            out['amp']=True
            out['scaler_state_dict_G']=self.scalerG.state_dict()
            out['scaler_state_dict_D']=self.scalerD.state_dict()

        torch.save(out,os.path.join(directory,model_name))

    def load_checkpoint(self,path):
        """Loads a model checkpoint, returns saved epoch."""
        checkpoint=torch.load(path)

        self.generator.load_state_dict(checkpoint['model_state_dict_G'])
        self.optimizerG.load_state_dict(checkpoint['optimizer_state_dict_G'])
        self.discriminator.load_state_dict(checkpoint['model_state_dict_D'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_state_dict_D'])
        
        self.amp = checkpoint['amp']
        self.iters = checkpoint['iters']

        if self.amp:
            self.scalerG.load_state_dict(checkpoint['scaler_state_dict_G'])
            self.scalerD.load_state_dict(checkpoint['scaler_state_dict_D'])
        
        return checkpoint['epoch']
        

    def generate_fake_images(self,batch_size):
        """Generates a batch of fake images."""
        noise = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)
        return self.generator(noise)

    def train_discriminator(self,images):
        """Trains discriminator on a single batch of images returns discriminator loss."""
        self.discriminator.zero_grad(set_to_none=True)

        requires_grad(self.generator,False)
        requires_grad(self.discriminator,True)


        #clamp parameters as per original WGAN paper
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01) 

        real_images = images.to(self.device)
        batch_size = real_images.size(0)
        

        #TODO Consider whether things should be inside or outside of AMP. 

        #Maximize D(x) - D(G(z))
        #Using softplus since I assume we want strictly positive values before the averaging operation? Probably doesn't matter
        if self.amp:
            with torch.cuda.amp.autocast():
                D_x = F.softplus(self.discriminator(real_images).view(-1)).mean()  #real loss. unsure if I should just put the negative inside the softplus?

                fake_images = self.generate_fake_images(batch_size)#.detach() #.detach() don't think i need this detach anymore due to settting requires grad
                D_g_z = F.softplus(self.discriminator(fake_images).view(-1)).mean() #fake loss
                loss_d=-(D_x - D_g_z)

            self.scalerD.scale(loss_d).backward()
            
            self.scalerD.step(self.optimizerD)
            self.scalerD.update()

        else:
            D_x = F.softplus(self.discriminator(real_images).view(-1)).mean() #real loss

            fake_images = self.generate_fake_images(batch_size)#.detach() don't think i need this detach anymore due to settting requires grad
            D_g_z = F.softplus(self.discriminator(fake_images).view(-1)).mean() #fake loss
            loss_d=-(D_x - D_g_z)

            loss_d.backward()
            self.optimizerD.step()
        
        loss_d = loss_d.detach()
        D_x, D_g_z = D_x.detach(), D_g_z.detach()

        self.iters +=1 #We only increment iteration number on training of Discriminator
        return loss_d, D_x, D_g_z
            
    def train_generator(self,batch_size):
        """Trains generator on a single batch and returns generator loss."""
        self.generator.zero_grad(set_to_none=True)

        requires_grad(self.generator,True)
        requires_grad(self.discriminator,False)

        #Generate fake images
        noise = torch.randn(batch_size, self.num_features, 1, 1, device=self.device)
        fake_images = self.generate_fake_images(batch_size)

        #Maximize D(G(z))
        if self.amp:
            with torch.cuda.amp.autocast():
                loss_g = -F.softplus(self.discriminator(fake_images).view(-1)).mean() #negative inside softplus?
            
            # Calculate gradients for G
            self.scalerG.scale(loss_g).backward()

            # Update G
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

        else:   
            loss_g = -F.softplus(self.discriminator(fake_images).view(-1)).mean()
            
            loss_g.backward()
            self.optimizerG.step()
        
        return loss_g
