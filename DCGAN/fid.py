import numpy as np
import torch, os, argparse, torchvision, torch.utils.data, time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import tqdm
import torch.optim as optim
from pytorch_fid import fid_score
from model import DCGAN

def compute_fid(path_to_scores,GAN,batch_size=128):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    num_images = (10000//batch_size +1) * batch_size
    pred_arr = np.empty((num_images, dims))

    start_idx = 0
    
    for i in tqdm(range(num_images)):
        with torch.no_grad():
            batch = GAN.generate_fake_images(batch_size)
            pred = model(batch)[0]

        ###Copied from fid_score module###
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu_2 = np.mean(pred_arr,axis=0)
    sigma_2 = np.cov(pred_arr,rowvar=False)
    with np.load(path_to_scores) as f:
        mu_1, sigma_1 = f['mu'][:], f['sigma'][:]

    return fid_score.calculate_frechet_distance(mu1,s1,mu2,s2)

def compute_fid_from_model_save(path_to_scores, path_to_model):
    GAN=DCGAN(1, 128, 4,latent_vector_size=128)
    GAN.load_checkpoint(path_to_model)
    return compute_fid(path_to_scores,GAN)