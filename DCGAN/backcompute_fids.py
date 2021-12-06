import os
import torch
from torch.utils.tensorboard import SummaryWriter
from fid import compute_fid_from_model_save
model_folder = "/home/ej74/checkpoints"
precomputed_path='home/ej74/128px.nz'
writer =SummaryWriter()
num_gpu = 1
batch_size = 300
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")
for path in os.listdir(model_folder):
    if path.endswith('.pt'):
       epoch=int(path.split('_model.pt')[0].split("epoch")[1])
       value= compute_fid_from_model_save(precomputed_path,os.path.join(model_folder,path),batch_size = batch_size)
       writer.add_scalar("FID",value,epoch)
       print(epoch,value)
