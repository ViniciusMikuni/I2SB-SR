import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import h5py
import numpy as np



class MinMaxNormalize:
    def __init__(self, min_values, max_values):
        self.min_values = torch.tensor(min_values).view(-1, 1, 1)
        self.max_values = torch.tensor(max_values).view(-1, 1, 1)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()
        return (x - self.min_values) / (self.max_values - self.min_values)

class NSKT(Dataset):
    def __init__(self, h5_file, transform=None,patch_size=256):
        self.h5_file = h5_file
        self.transform = transform
        self.patch_size = patch_size
        #The low resolution images are created by the super resolution model during training

        with h5py.File(h5_file, 'r') as hf:
            # Get the shape of the dataset
            u_shape = hf['u'].shape
            v_shape = hf['v'].shape
            w_shape = hf['w'].shape
            self.data_shape = (u_shape[0], u_shape[1] + v_shape[1] + w_shape[1], u_shape[2], u_shape[3])
        self.num_patches_per_image = (self.data_shape[2] // patch_size) * (self.data_shape[3] // patch_size)
        
    def __len__(self):
        return self.data_shape[0]* self.num_patches_per_image
    
    def __getitem__(self, idx):
        image_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image
        patch_row = (patch_idx // (self.data_shape[3] // self.patch_size)) * self.patch_size
        patch_col = (patch_idx % (self.data_shape[3] // self.patch_size)) * self.patch_size

        with h5py.File(self.h5_file, 'r') as hf:
            u = hf['u'][image_idx]
            v = hf['v'][image_idx]
            w = hf['w'][image_idx]
            image = np.concatenate((u, v, w), axis=0)

        patch = image[:, patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        patch = torch.tensor(patch, dtype=torch.float32)        

        if self.transform:
            image = self.transform(patch)
        return image


def get_data_loader(file_path):

    # Define transformations

    #Calculated from the 16000_2048_2048_seed_3407_uvw.h5 file! change it in case it is needed    
    min_values = [-3.09785827, -3.59914398, -74.59442477]
    max_values = [3.74879445, 3.71856606, 67.37008203]
    min_max_normalize = MinMaxNormalize(min_values, max_values)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(min_max_normalize),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    
    dataset = NSKT(h5_file=file_path,transform=transform)
    return dataset
