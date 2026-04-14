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

# based on Vonderfecht, 2025
SCHEDULER = [999, 972, 949, 929, 897, 869, 834, 805, 780, 751, 726, 704, 688, 670, 648, 627, 608, 591, 578, 561,
             546, 530, 520, 510, 498, 491, 480, 465, 455, 447, 438, 429, 419, 410, 402, 390, 380, 371, 361, 353,
             345, 336, 326, 319, 313, 305, 296, 289, 282, 276, 269, 261, 254, 247, 242, 237, 231, 224, 219, 213,
             209, 204, 200, 194, 189, 185, 181, 175, 170, 167, 163, 160, 156, 153, 149, 146, 143, 139, 135, 132,
             129, 125, 121, 118, 116, 113, 110, 107, 104, 101, 99, 96, 94, 92, 90, 87, 85, 82, 80, 78, 76, 74, 72,
             70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
             34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
             9, 8, 7, 6, 5, 4, 3, 2, 1]

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

def evenly_spaced(lst, x):
    if x <= 0:
        return []
    if x >= len(lst):
        return lst

    n = len(lst)
    step = (n - 1) / (x - 1)

    indices = [round(i * step) for i in range(x)]
    return [lst[i] for i in indices]


def get_no_bits_steps(T, K, M, C, H, W):
    bpp = turbo_ddcm_bpp(T, K, M, C, 0, H, W) # assuming NBS (no bits steps) is 0

    if H * W <= 768 ** 2:
        param = 70 # heuristic
    else:
        param = 20 # heuristic

    bins = torch.logspace(
        start=torch.log10(torch.tensor(0.01)), end=torch.log10(torch.tensor(0.15)), steps=param
    )

    nbs = torch.max(torch.tensor(0),
                    torch.min(torch.tensor(T - 2),
                              # NBS <= T - 2 (we have to do one step with bits and we have one DDIM step
                              # either way at the end of the DDPM process).
                              param - torch.bucketize(bpp, bins) - 1)).item()

    bpp = turbo_ddcm_bpp(T, K, M, C, nbs, H, W)

    return nbs, bpp


# ----- Flow Matching (based on Vonderfecht, 2025) ------

def get_ot_flow_to_ddpm_factor(snr):
    OT_flow_noise_sigma = 1 / (snr + 1)

    alpha_cumprod = snr ** 2 / (snr ** 2 + 1)
    DDPM_noise_sigma = torch.sqrt(1 - alpha_cumprod)

    ot_flow_to_ddpm_factor = DDPM_noise_sigma / OT_flow_noise_sigma

    return ot_flow_to_ddpm_factor

def sigma_to_snr(sigma):
    return (1 - sigma) / sigma

def get_alpha_prod_and_beta_prod(snr):
    if snr == torch.inf:
        alpha_prod = 1
    else:
        alpha_prod = snr ** 2 / (1 + snr ** 2)
    beta_prod = 1 - alpha_prod
    return alpha_prod, beta_prod
