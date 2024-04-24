import os

import numpy as np
import yaml
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from torchvision import transforms
from PIL import Image

from dragonfruitvp.src.evaluator import DragonFruitEvaluate
from dragonfruitvp.utils.parser import create_parser, default_parser

class CustomDataset(Dataset):
    def __init__(self, directory, normalize=False, data_name='custom'):
        super().__init__()
        self.video_folders = [f for f in sorted(Path(directory).iterdir()) if f.is_dir()]

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

        self.transform_mask = lambda x: torch.from_numpy(x).long()

    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, index):
        files = sorted(self.video_folders[index].glob('*.png'), key=lambda x: int(x.stem.split('_')[-1]))
        video_frames = []
        for file in files:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            ])
            # pil_image = Image.open(file).convert('RGB')
            pil_image = Image.open(file)
  
            video_frames.append(transform(pil_image))

        data = torch.stack(video_frames[:11]).float()
        # label = torch.stack(video_frames[11:]).float()
        masks_dir = self.video_folders[index].joinpath("mask.npy")
        mask = self.transform_mask(np.load(masks_dir)[-1]) if masks_dir.exists() else []


        return data, mask





if __name__ == "__main__":
    args = create_parser().parse_args()
    config = vars(args)

    with open(config['model_config_file']) as model_config_file:
        custom_model_config = yaml.safe_load(model_config_file)
    
    with open(config['training_config_file']) as training_config_file:
        custom_training_config = yaml.safe_load(training_config_file)

    custom_training_config['batch_size'] = 8

    # Define some hyper parameters
    BATCH_SIZE=custom_training_config['batch_size']

    train_set = CustomDataset('dataset/train')
    val_set = CustomDataset('dataset/val')
    hidden_set = CustomDataset('dataset/hidden')

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1
    )

    dataloader_hidden = torch.utils.data.DataLoader(
        hidden_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1
    )    

    print('dataloader type', type(dataloader_val))

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    # config['res_dir'] = 'pretrainedVP_weights'
    config['ex_name'] = config['model_config_file'][:-5].split('/')[-1] + '_' + config['training_config_file'][:-5].split('/')[-1]
    print(config['ex_name'], type(config['ex_name']))

    config['vp_weight'] = os.path.join(config['res_dir'], config['ex_name'], 'checkpoints', 'best.ckpt')
    config['unet_weight'] = os.path.join(config['res_dir'],'unet', 'best_model.pth')

    exp = DragonFruitEvaluate(args, dataloaders=(dataloader_train, dataloader_val, dataloader_hidden), strategy='auto')

    print('>'*35 + ' validating ' + '<'*35)
    exp.val()

    # print('>'*35 + 'testing' + '<'*35)
    # results = exp.test()
    # print(results.shape)
    # file_path = "submission.pt"
    # torch.save(results, file_path)
