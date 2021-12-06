import torch, os, torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utilities import weights_init

class GaussianNoise(nn.Module):
    """Applies gaussian noise to an image input."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self,image,):
        batch, _, height, width =image.size()
        noise = image.new_empty(batch,1,height,width).normal_()
        return image + self.weight * noise

class Generator(nn.Module):
    """Generator network for DCGAN."""
    def __init__(self,ngpu, number_generated_features, n_convolution_blocks,latent_vector_size = 128,num_channels=3):
        super().__init__()
        self.ngpu = ngpu
        
        input_layer_size = number_generated_features * 2**(n_convolution_blocks)
        
        #first convolute latent vector Z 
        modules = [ nn.ConvTranspose2d(latent_vector_size, input_layer_size, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(input_layer_size),
                    nn.LeakyReLU(0.2, inplace=True),
                    GaussianNoise()] 
        
        for i in range(n_convolution_blocks):
            modules+=self.ConvolutionBlock(input_layer_size,input_layer_size//2)
            input_layer_size//=2


        modules += [nn.ConvTranspose2d(number_generated_features, num_channels, 4, 2, 1, bias=False),
                    nn.Tanh()]   

        self.main = nn.Sequential(*modules)

    def ConvolutionBlock(self, in_channels, out_channels,dropout =True, dropout_value = 0.5):
        """Returns a convolutional upscaling block.
        
        Args: TO DO
        """
        if dropout:
            return [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Dropout2d(dropout_value),
                            GaussianNoise()
                            ]
        
        else:
            return [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            GaussianNoise()] 
        
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """Discriminator network for DCGAN."""
    def __init__(self,ngpu, num_features, n_convolution_blocks,num_channels=3,sigmoid =True):
        super().__init__()
        self.ngpu = ngpu

        modules = self.ConvolutionBlock(num_channels,num_features)
        input_layer_size = num_features
        for i in range(n_convolution_blocks):
            modules+=self.ConvolutionBlock(input_layer_size,input_layer_size*2)
            input_layer_size*=2
        
        modules += [nn.Conv2d(input_layer_size, 1, 2, stride=4, padding=0, bias=False)]
        
        #Optional sigmoid output since Sigmoid is not compatible with AMP when using BCE as loss
        if sigmoid: 
            modules+=[nn.Sigmoid()]
        

        self.main = nn.Sequential(*modules)
        
        
    def ConvolutionBlock(self, in_channels, out_channels,dropout =True, dropout_value = 0.5):
        """Returns a convolutional downscaling block.
        
        Args: TO DO
        """
        if dropout:
            return [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout_value)]
        
        else:
            return [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)]
    
    def forward(self, input):
        return self.main(input)


