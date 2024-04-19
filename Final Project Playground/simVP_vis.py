import torch
from PIL import Image

import numpy as np
from torchvision import transforms
from openstl.utils import show_video_line

def convert_img(array):
    # array_transposed = np.transpose(array, (1, 2, 0))
    transform = transforms.ToPILImage()
    pil_image = transform(torch.tensor(array))

    return pil_image

def test_load(file_path):
    transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
    pil_image = Image.open(file_path)
    torch_tensor = transform(pil_image)
    print(torch_tensor.shape)

    transform2 = transforms.ToPILImage()
    pil_image2 = transform2(torch_tensor)
    pil_image2.save('test.png')


if __name__ == "__main__":
    # # show the given frames from an example
    inputs = np.load('./work_dirs/custom_exp/saved/inputs.npy')
    preds = np.load('./work_dirs/custom_exp/saved/preds.npy')
    trues = np.load('./work_dirs/custom_exp/saved/trues.npy')

    example_idx = 0

    true_img = trues[example_idx][-1]
    img = convert_img(true_img)
    img.save('exp_true.png')

    pred_img = preds[example_idx][-1]
    img = convert_img(pred_img)
    img.save('exp_pred.png')

    # test_load('/scratch/yg2709/Deep-Learning/Final Project Playground/dataset/train/video_00000/image_0.png')