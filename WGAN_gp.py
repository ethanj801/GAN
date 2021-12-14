import torch, os, torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import Discriminator, Generator
from utilities import weights_init, requires_grad
from torch.nn import functional as F
from WGAN import WGAN

class WGAN_GP(WGAN):
    """Complete WGAN model.
    
    Attributes: #TODO

    """

    def __init__(self,num_gpu, num_features, n_convolution_blocks,lr = 0.00005,
                 latent_vector_size=128, AMP = False,lambda_gp = 10):
        """Constructor for DCGAN Class.
        
        Args:
        lambda_gp: weight for the gradient penalty


        """
        super().__init__(num_gpu, num_features,n_convolution_blocks,lr=lr, latent_vector_size=latent_vector_size, AMP=AMP)
        self.lambda_gp=lambda_gp

    def compute_gradient_penalty(self,real_samples, fake_samples): 
        #taken from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), device = self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        #fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake =  torch.ones(real_samples.shape[0], 1, device = self.device, requires_grad =False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def train_discriminator(self,images):
        """Trains discriminator on a single batch of images returns discriminator loss."""
        self.discriminator.zero_grad(set_to_none=True)

        requires_grad(self.generator,False)
        requires_grad(self.discriminator,True)


        real_images = images.to(self.device)
        batch_size = real_images.size(0)
        

        #TODO Consider whether things should be inside or outside of AMP. 

        #Maximize D(x) - D(G(z))
        #Using softplus since I assume we want strictly positive values before the averaging operation? Probably doesn't matter. 
        #
        if self.amp:
            with torch.cuda.amp.autocast():
                critic_score_real=self.discriminator(real_images).view(-1)
                real_loss = F.softplus(-critic_score_real).mean() 
                

                fake_images = self.generate_fake_images(batch_size)#.detach() #.detach() don't think i need this detach anymore due to settting requires grad
                critic_score_fake = self.discriminator(fake_images).view(-1)
                fake_loss = F.softplus(critic_score_fake).mean()

                penalty =self.compute_gradient_penalty(real_images,fake_images)
                loss_d = real_loss+fake_loss + penalty *self.lambda_gp
                
            self.scalerD.scale(loss_d).backward()

            self.scalerD.step(self.optimizerD)
            self.scalerD.update()

        else:
            critic_score_real=self.discriminator(real_images).view(-1)
            real_loss = F.softplus(-critic_score_real).mean() 

            fake_images = self.generate_fake_images(batch_size)#.detach() #.detach() don't think i need this detach anymore due to settting requires grad
            critic_score_fake = self.discriminator(fake_images).view(-1)
            fake_loss = F.softplus(critic_score_fake).mean()
            
            penalty =self.compute_gradient_penalty(real_images,fake_images)
            loss_d = real_loss+fake_loss + penalty *self.lambda_gp

            loss_d.backward()
            self.optimizerD.step()
        
        loss_d = loss_d.detach()
        critic_score_real = critic_score_real.mean().detach() 
        critic_score_fake =critic_score_fake.mean().detach()
        
        self.iters +=1 #We only increment iteration number on training of Discriminator
        return loss_d, critic_score_real, critic_score_fake
    

    def set_learning_rate_D(self,lr):
        """Updates the learning rate of the critic optimizer. Useful as when models are loaded optimizer learning rates are overwritten"""
        for group in self.optimizerD.param_groups:
            group['lr'] = lr

    def set_learning_rate_G(self,lr):
        """Updates the learning rate of the generator optimizer. Useful as when models are loaded optimizer learning rates are overwritten"""
        for group in self.optimizerG.param_groups:
            group['lr'] = lr
            
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
                loss_g = F.softplus(-self.discriminator(fake_images).view(-1)).mean()
            
            # Calculate gradients for G
            self.scalerG.scale(loss_g).backward()

            # Update G
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

        else:   
            loss_g = F.softplus(-self.discriminator(fake_images).view(-1)).mean()
            
            loss_g.backward()
            self.optimizerG.step()
        
        return loss_g
