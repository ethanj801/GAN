import os
from torch.utils.tensorboard import SummaryWriter
from fid import compute_fid_from_model_save
model_folder = "home/ej74/checkpoints"
precomputed_path='home/ej74/128px.nz'
writer =SummaryWriter()
for path in os.listdir(model_folder):
    if path.endswith('.pt'):
        compute_fid_from_model_save(precomputed_path,path)