import torch
import os
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
from dragonfruitvp.src.simvp import SimVP_Model, SimVP
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

def ssim_test():

    dimensions = [3, 160, 240]
    pred = np.random.rand(*dimensions).astype(np.float32)
    true = np.random.rand(*dimensions).astype(np.float32)
    print(cal_ssim(pred, true, multichannel=True, channel_axis=0, data_range=1))


def load_model_test():
    model = SimVP(
            in_shape=[11, 3, 160, 240]
            # steps_per_epoch=len(self.data.train_loader),
            # test_mean = self.data.test_mean,
            # test_std = self.data.test_std,
            # save_dir = save_dir, **self.config
        )
    checkpoint_path = "./work_dirs/Debug/checkpoints/best.ckpt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(model)




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
            pil_image = Image.open(file).convert('RGB')
  
            video_frames.append(transform(pil_image))

        data = torch.stack(video_frames[:11]).float()
        label = torch.stack(video_frames[11:]).float()
        masks_dir = self.video_folders[index].joinpath("mask.npy")
        mask = self.transform_mask(np.load(masks_dir)[-1]) if masks_dir.exists() else None


        return data, label, mask

def test_dataset():
    BATCH_SIZE=8

    base_dir = 'testing_outputs'
    os.makedirs(base_dir, exist_ok=True)

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

    for batch, (datas, labels, masks) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
        frames = labels[:, -1, :, :, :].squeeze(1)
        # print(frames.shape)
        # print(masks.shape)
        save_dir = os.path.join(base_dir, f'batch{batch}')
        os.makedirs(save_dir, exist_ok=True)

        for i in range(frames.shape[0]):
            # print(type(frames[i]), frames[i].shape)
            frame_true = to_pil_image(frames[i])
            mask_true = to_pil_image(masks[i].byte().data)

            plt.imsave(os.path.join(save_dir, f'exp_frame_{i}.png'), np.array(frame_true))
            plt.imsave(os.path.join(save_dir, f'exp_mask_{i}.png'), mask_true) 

            
            # frame_true.save(os.path.join(save_dir, f'exp_frame_{i}.png'))
            # mask_true.save(os.path.join(save_dir, f'exp_mask_{i}.png')) 

if __name__ == "__main__":
    # load_model_test()
    test_dataset()