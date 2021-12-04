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
            modules+=ConvolutionBlock(input_layer_size,input_layer_size//2)
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


class DCGAN():
    """Complete DCGAN model.
    
    Attributes: #TODO

    """

    def __init__(self,num_gpu, num_features, n_convolution_blocks,lr = 0.0002, beta1 = 0.5,
                 real_label_value = 0.9, fake_label_value = 0, latent_vector_size=128, AMP = False):
        """Constructor for DCGAN Class.
        
        Args:
            num_gpu: Number of GPUs to use.
            num_features: Size of the image.
            n_convolution_blocks: Number of convolutional block in Generator and Discriminator.
            beta1: Beta1 hyperparameter for Adam optimizers.
            lr: Learning rate for optimizers.
            real_label_value: Value to assign real labels during training.
            fake_label_value: Value to assign fake labels during training.
            latent_vector_size: Size of latent vector for Generator
            AMP: When True, enables Automatic Mixed Precision.


        """
        self.num_gpu = num_gpu
        self.num_features = num_features
        self.real_label =real_label_value
        self.fake_label_value = fake_label_value
        self.latent_vector_size=latent_vector_size
        self.amp = AMP
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.iters = 0 #number of discriminator training iterations

        #Set up models. 
        if self.amp:
            self.scalerD = torch.cuda.amp.GradScaler()
            self.scalerG = torch.cuda.amp.GradScaler()
            self.criterion = nn.BCEWithLogitsLoss()

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_block,sigmoid=False).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = 128)

        else:
            self.scalerD = None
            self.scalerG = None
            self.criterion =nn.BCELoss()

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_block,sigmoid=True).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = 128)


        #Multi GPU compatibility -- untested as of Dec 4 2021
        if self.device.type == 'cuda' and self.num_gpu > 1:
            self.discriminator = nn.DataParallel(self.discriminator, list(range(self.num_gpu)))
            self.generator = nn.DataParallel(self.generator, list(range(self.num_gpu)))
        
        #Initialize weights as per DCGAN paper
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        #Set up optimizers 
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))

    def save_checkpoint(self,directory,epoch = None,model_name = 'model.pt'):
        """Saves a model checkpoint."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        out = {'epoch': epoch,
            'model_state_dict_G': self.generator.state_dict(),
            'optimizer_state_dict_G': self.optimizerG.state_dict(),
            'model_state_dict_D': self.Discriminator.state_dict(),
            'optimizer_state_dict_D': self.optimizerD.state_dict(),
            'iters': iters
            'amp': False #changed below if amp is enabled
            }
        if self.amp:
            out['amp']=True
            out['scaler_state_dict_G']=self.scalerG.state_dict()
            out['scaler_state_dict_D']=self.scalerD.state_dict()

        torch.save(os.path.join(directory,model_name))

    def load_checkpoint(path):
        """Loads a model checkpoint, returns saved epoch."""
        checkpoint=torch.load(path)

        self.generator.load_state_dict(checkpoint['model_state_dict_G'])
        self.optimizerG.load_state_dict(checkpoint['optimizer_state_dict_G'])
        self.Discriminator.load_state_dict(checkpoint['model_state_dict_D'])
        self.optimizerD.load_state_dict(checkpoint['optimizer_state_dict_D'])
        
        self.amp = checkpoint['amp']
        self.iters = checkpoint['iters']

        if self.amp:
            self.scalerG.load_state_dict(checkpoint['scaler_state_dict_G'])
            self.scalerD.load_state_dict(checkpoint['scaler_state_dict_D'])
        
        return checkpoint['epoch']
        

    def generate_fake_images(self,batch_size):
        """Generates a batch of fake images."""
        noise = torch.randn(batch_size, self.num_features, 1, 1, device=self.device)
        return self.generator(noise)

    def train_discriminator(self,images):
        """Trains discriminator on a single batch of images returns discriminator loss."""
        self.discriminator.zero_grad()

        real_images = images.to(self.device)
        batch_size = real_images.size(0)
        

        #Maximize log(D(x)) + log(1 - D(G(z)))
        label = torch.full((batch_size,), self.real_label_value, dtype=torch.float, device=self.device)
        if self.amp:
            with torch.cuda.amp.autocast():
                output = self.discriminator(real_images).view(-1)
                errD_real = self.criterion(output, label)  

            self.scalerD.scale(errD_real).backward()
            
            label = torch.full((batch_size,), self.fake_label_value, dtype=torch.float, device=self.device)
            fake_images = self.generate_fake_images(batch_size).detach() #gradient not needed for generator here, hence detach
            
            with torch.cuda.amp.autocast():
                output = self.discriminator(fake_images.view(-1))
                errD_fake = self.criterion(output, label)
            
            self.scalerD.scale(errD_fake).backward()
            self.scalerD.step(self.optimizerD)
            self.scalerD.update()

            errD = errD_real.item() + errD_fake.item()

        else:
            output = self.discriminator(real_images).view(-1)
            errD_real = self.criterion(output, label)  
            errD_real.backward()

            label = torch.full((batch_size,), self.fake_label_value, dtype=torch.float, device=self.device)
            fake_images = self.generate_fake_images(batch_size).detach() #gradient not needed for generator here, hence detach

            output = self.discriminator(fake_images.view(-1))
            errD_fake = self.criterion(output, label)

            self.optimizerD.step()

            errD = errD_real.item() + errD_fake.item()

        self.iters +=1 #We only increment iteration number on training of Discriminator
        return errD
            
    def train_generator(self,batch_size):
        """Trains generator on a single batch and returns generator loss."""
        self.generator.zero_grad()

        #Generate fake images
        noise = torch.randn(batch_size, self.num_features, 1, 1, device=self.device)
        fake_images = self.generate_fake_images(batch_size)

        #using 1.0 here instead of the real label value. 
        #TODO: Test the relative performance of this
        label_value = 1.0 

        #Generating label vector
        #QUESTION: Will it provide meaningful performance upgrades if I keep the I be keeping this variable on device?
        #Downside would be that batch_size wouldn't be as easy to adjust on the fly. I assume it is negligible? 
        label = torch.full((batch_size,), label_value, dtype=torch.float, device=self.device)

        #Maximize log(D(G(z)))
        if self.amp:
            with torch.cuda.amp.autocast():
                output = self.discriminator(fake).view(-1)
                errG = self.criterion(output, label)
            
            # Calculate gradients for G
            self.scalerG.scale(errG).backward()

            # Update G
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

        else:   
            output = self.discriminator(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            self.optimizerG.step()
        
        return errG
