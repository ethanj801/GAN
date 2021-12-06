import torch.nn as nn
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def weights_init(m):
    """Initialize model weights to a normal distribution with mean=0, stdev=0.02."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
