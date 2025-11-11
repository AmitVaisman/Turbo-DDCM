import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

BIN_SUFFIX = '.turbo_ddcm'

def save_as_binary(encoding, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Convert bitstring to bytes **including padding** in little endian
    byte_array = int(encoding, 2).to_bytes((len(encoding) + 7) // 8, byteorder='big')
    # Write to binary file
    with open(filename, 'wb') as f:
        f.write(byte_array)

def load_binary(filename):
    with open(filename, 'rb') as f:
        byte_data = f.read()

    bitstring = bin(int.from_bytes(byte_data, byteorder='big'))[2:]  # Remove '0b' prefix
    bitstring = bitstring.zfill(len(byte_data) * 8)  # pad with zeros to full byte length
    return bitstring

def down_sample_mask(mask, kernel_size, device):
    mask = F.avg_pool2d(mask, kernel_size=kernel_size)
    mask = mask.repeat(1, 4, 1, 1).to(device)
    return mask
            
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(image_path, resize_to, device=None):
    class MinusOneToOne(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * 2 - 1

    class ResizePIL(torch.nn.Module):
        def __init__(self, image_size):
            super().__init__()
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            self.image_size = image_size

        def forward(self, pil_image: Image.Image) -> Image.Image:
            if self.image_size is not None and pil_image.size != self.image_size:
                pil_image = pil_image.resize(self.image_size)
            return pil_image


    image = Image.open(image_path).convert('RGB')
    transforms_ = transforms.Compose([ResizePIL(resize_to), transforms.ToTensor(), MinusOneToOne()])
    image = transforms_(image)

    return image.unsqueeze(0).to(device)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)

    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.detach().cpu().squeeze().numpy()
    if x.ndim == 3:
        return np.transpose(x, (1, 2, 0))
    else:
        return x

def turbo_ddcm_bpp(T, K, M, C, NBS, img_height, img_width):
    # (T - 1) since there is no noise addition on last step
    bits = (T - NBS - 1) * (math.ceil(math.log2(math.comb(K, M))) + M * C)
    bpp = bits / (img_height * img_width)
    return round(bpp, 8)

def save_decoded_img(filename, w_dec):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.imsave(filename, clear_color(w_dec))
