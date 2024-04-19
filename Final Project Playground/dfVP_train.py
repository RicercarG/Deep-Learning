import time
import torch
import os
import numpy as np
import pickle
import argparse
import yaml

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from dragonfruitvp.src.trainer import DragonFruitExperiment
from dragonfruitvp.utils.parser import create_parser, default_parser


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time for {func.__name__} is {end_time-start_time}")
        return result
    return wrapper


class CustomDataset(Dataset):
    def __init__(self, directory, normalize=False, data_name='custom', limit=-1):
        super().__init__()
        self.video_folders = [f for f in sorted(Path(directory).iterdir()) if f.is_dir() ][:limit]
        self.data_name = data_name
        self.mean = None
        self.std = None

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

    args = create_parser().parse_args()
    config = vars(args)

    with open(config['model_config_file']) as model_config_file:
        custom_model_config = yaml.safe_load(model_config_file)
    
    with open(config['training_config_file']) as training_config_file:
        custom_training_config = yaml.safe_load(training_config_file)

    # Define some hyper parameters
    BATCH_SIZE=custom_training_config['batch_size']

    # load dataset
    limit = -1
    train_set = CustomDataset('dataset/train', limit=limit)
    val_set = CustomDataset('dataset/val', limit=limit)
    unlabeled_set = CustomDataset('dataset/unlabeled', limit=limit)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )

    dataloader_unlabeled = torch.utils.data.DataLoader(
        unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )




    # update default parameters
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)

    exp = DragonFruitExperiment(args, dataloaders=(dataloader_unlabeled, dataloader_val, dataloader_train), strategy='auto')

    print('>'*35 + ' training ' + '<'*35)
    exp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    exp.test()