from models import Discriminator, Generator
class WGAN():
    """Complete WGAN model.
    
    Attributes: #TODO

    """

    def __init__(self,num_gpu, num_features, n_convolution_blocks,lr = 0.0002,
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
            self.criterion = nn.BCEWithLogitsLoss()

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_blocks,sigmoid=False).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = 128).to(self.device)

        else:
            self.scalerD = None
            self.scalerG = None
            self.criterion =nn.BCELoss()

            self.discriminator =Discriminator(self.num_gpu, num_features, n_convolution_blocks,sigmoid=True).to(self.device)
            self.generator = Generator(self.num_gpu, num_features, n_convolution_blocks,latent_vector_size = 128).to(self.device)


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
                output = self.discriminator(fake_images).view(-1)
                errD_fake = self.criterion(output, label)
            
            self.scalerD.scale(errD_fake).backward()
            self.scalerD.step(self.optimizerD)
            self.scalerD.update()

            errD = errD_real.detach() + errD_fake.detach()

        else:
            output = self.discriminator(real_images).view(-1)
            errD_real = self.criterion(output, label)  
            errD_real.backward()

            label = torch.full((batch_size,), self.fake_label_value, dtype=torch.float, device=self.device)
            fake_images = self.generate_fake_images(batch_size)#.detach() #gradient not needed for generator here, hence detach
            output = self.discriminator(fake_images.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()

            self.optimizerD.step()

            errD = errD_real.detach() + errD_fake.detach()
        #self.fake = fake_images
        self.iters +=1 #We only increment iteration number on training of Discriminator
        return errD
            
    def train_generator(self,batch_size):
        """Trains generator on a single batch and returns generator loss."""
        self.generator.zero_grad(set_to_none=True)

        #Generate fake images
        noise = torch.randn(batch_size, self.num_features, 1, 1, device=self.device)
        fake_images = self.generate_fake_images(batch_size)

        #using 1.0 here instead of the real label value. 
        #TODO: Test the relative performance of this
        label_value = 1.0#self.real_label_value 

        #Generating label vector
        #QUESTION: Will it provide meaningful performance upgrades if I keep the I be keeping this variable on device?
        #Downside would be that batch_size wouldn't be as easy to adjust on the fly. I assume it is negligible? 
        label = torch.full((batch_size,), label_value, dtype=torch.float, device=self.device)

        #Maximize log(D(G(z)))
        if self.amp:
            with torch.cuda.amp.autocast():
                output = self.discriminator(fake_images).view(-1)
                errG = self.criterion(output, label)
            
            # Calculate gradients for G
            self.scalerG.scale(errG).backward()

            # Update G
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

        else:   
            output = self.discriminator(fake_images).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            self.optimizerG.step()
        
        return errG.detach()

