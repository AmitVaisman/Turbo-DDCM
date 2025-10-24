import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
# from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.transforms.functional import to_tensor

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


def load_image(image_path, device=None):
    class MinusOneToOne(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * 2 - 1
        
    image = Image.open(image_path).convert('RGB')
    transforms_ = transforms.Compose([transforms.ToTensor(), MinusOneToOne()])
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

# def compute_psnr(gt_img_path, rec_img_path, device, num_patches=1, save_annotated_path=None):
#     PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
#
#     gt_img = to_tensor(Image.open(gt_img_path).convert('RGB')).to(device).unsqueeze(0)
#     rec_img = to_tensor(Image.open(rec_img_path).convert('RGB')).to(device).unsqueeze(0)
#
#     _, _, H, W = gt_img.shape
#     patch_H = H // num_patches
#     patch_W = W // num_patches
#
#     psnr_values = []
#
#     # Convert one of the images to PIL for annotation
#     annotated_img = Image.open(gt_img_path).convert("RGB")
#     draw = ImageDraw.Draw(annotated_img)
#
#     # Try loading a font (fallback to default if unavailable)
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=patch_H // 5)
#     except IOError:
#         font = ImageFont.load_default()
#
#     for i in range(num_patches):
#         for j in range(num_patches):
#             h_start = i * patch_H
#             h_end = h_start + patch_H
#             w_start = j * patch_W
#             w_end = w_start + patch_W
#
#             gt_patch = gt_img[:, :, h_start:h_end, w_start:w_end]
#             rec_patch = rec_img[:, :, h_start:h_end, w_start:w_end]
#
#             psnr = PSNR(gt_patch, rec_patch).item()
#             psnr_values.append(round(psnr, 2))
#
#             if save_annotated_path:
#                 # Draw rectangle and PSNR text
#                 draw.rectangle([(w_start, h_start), (w_end, h_end)], outline="red", width=2)
#                 text = f"{psnr:.1f}"
#                 draw.text((w_start + 5, h_start + 5), text, fill="red", font=font)
#
#     if save_annotated_path:
#         annotated_img.save(save_annotated_path)
#         print(f"Saved annotated image to {save_annotated_path}")
#
#     all_img_psnr = round(PSNR(gt_img, rec_img).item(), 2)
#
#     return psnr_values, all_img_psnr