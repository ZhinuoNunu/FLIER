import torch
import numpy as np
import pandas as pd
from einops import rearrange

from PIL import Image

# conda activate ldm

# cache_feature = torch.load('/home/prp2/zzn/stable_diffusion/stablediffusion/outputs/cache_features/sd_samples_oxfordpets.pt')
cache_feature = torch.load('/home/prp2/zzn/stable_diffusion/stablediffusion/outputs/cache_features/sd_samples_oxfordpets.pt')
# print(len(cache_feature))
data = pd.read_pickle('/home/prp2/zzn/data/oxford_pets/oxfordpets_train_sd.pkl')
print(data.head())
image = data['image'].to_list()
cache = data['cache'].to_list()

target = []
for i in range(len(image)):
    if 'birman_00060.png' in image[i]:
        target.append(cache[i])
        break

print(image[i])
x = target[0][0]
y = target[0][1]
x_sample_0 = cache_feature[y][x]
# x_sample_0 = torch.reshape(x_sample_0, (-1,x_sample_0.shape[-1]))
x_sample_0 = 255. * rearrange(x_sample_0.cpu().numpy(), 'c h w -> h w c')
img = Image.fromarray(x_sample_0.astype(np.uint8))
img.save('./birman_nox.png')

# x_sample = cache_feature[0]
# x_sample_0 = x_sample[0]
# print(x_sample_0.shape)  # torch.Size([4, 64, 64])
# x_sample_0 = torch.reshape(x_sample_0, (-1,x_sample_0.shape[-1]))
# print(x_sample_0.shape)
cahce_set = tuple([cache_feature[i][0] for i in range(20)])
cache_list = torch.stack(cahce_set, dim=0)
print(cache_list.shape)
a = torch.stack((cache_feature[0][0], cache_feature[0][0],cache_feature[0][0],cache_feature[0][0]), dim=0)
print(a.shape)
# x_sample_0 = 255. * rearrange(x_sample_0.cpu().numpy(), 'c h w -> h w c')
# img = Image.fromarray(x_sample_0.astype(np.uint8))
# img.save('./test_nox.png')














