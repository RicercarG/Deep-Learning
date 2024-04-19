import torch
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

dimensions = [3, 160, 240]
pred = np.random.rand(*dimensions).astype(np.float32)
true = np.random.rand(*dimensions).astype(np.float32)

print(cal_ssim(pred, true, multichannel=True, channel_axis=0, data_range=1))
