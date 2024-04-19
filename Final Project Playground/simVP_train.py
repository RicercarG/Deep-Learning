import time
import torch
import os
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time for {func.__name__} is {end_time-start_time}")
        return result
    return wrapper


class CustomDataset(Dataset):
    def __init__(self, directory, normalize=False, data_name='custom'):
        super().__init__()
        self.video_folders = [f for f in sorted(Path(directory).iterdir()) if f.is_dir() ]
        self.mean = None
        self.std = None
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.video_folders)
 
    def __getitem__(self, index):
        def extract_number(s):
            return int(str(s).split('_')[-1].rstrip('.png'))
        files = sorted(self.video_folders[index].glob('*.png'), key=extract_number)
        video_frames = []
        for file in files:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            ])
            pil_image = Image.open(file)
  
            video_frames.append(transform(pil_image))

        data = torch.stack(video_frames[:11]).float()
        labels = torch.stack(video_frames[11:]).float()

        return data, labels
    

if __name__ == "__main__":
    # Define some hyper parameters
    BATCH_SIZE=16
    PRE_SEQ_LENGTH=11
    AFT_SEQ_LENGTH=11

    # load dataset
    train_set = CustomDataset('dataset/train')
    val_set = CustomDataset('dataset/val')
    unlabeled_set = CustomDataset('dataset/unlabeled')

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )

    dataloader_unlabeled = torch.utils.data.DataLoader(
        unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )
    
    # config for training
    custom_training_config = {
        'pre_seq_length': PRE_SEQ_LENGTH,
        'aft_seq_length': AFT_SEQ_LENGTH,
        'total_length': PRE_SEQ_LENGTH + AFT_SEQ_LENGTH,
        'batch_size': BATCH_SIZE,
        'val_batch_size': BATCH_SIZE,
        'epoch': 5,
        'lr': 0.0001,
        'metrics': ['mse', 'mae'],

        'ex_name': 'lr1e4',
        'dataname': 'custom',
        'in_shape': [11, 3, 160, 240],
    }

    custom_model_config = {
        # For MetaVP models, the most important hyperparameters are:
        # N_S, N_T, hid_S, hid_T, model_type
        'method': 'SimVP',
        # Users can either using a config file or directly set these hyperparameters
        # 'config_file': 'configs/custom/example_model.py',

        # Here, we directly set these parameters
        'model_type': 'gSTA',
        'N_S': 4,
        'N_T': 8,
        'hid_S': 64,
        'hid_T': 256
    }

    args = create_parser().parse_args([])
    config = args.__dict__

    # update default parameters
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)

    exp = BaseExperiment(args, dataloaders=(dataloader_unlabeled, dataloader_val, dataloader_train), strategy='auto')

    print('>'*35 + ' training ' + '<'*35)
    exp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    exp.test()