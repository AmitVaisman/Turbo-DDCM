import os
import argparse
import torch
import pandas as pd
import time
from tqdm import tqdm
import json

from turbo_ddcm.turbo_ddcm import TurboDDCM
from turbo_ddcm import utils

def main(args):
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    files = sorted(os.listdir(args.input_dir))
    target_files = [f for f in files if f.endswith('png')]
    if not target_files:
        raise FileNotFoundError(f"No files with suffix {args.img_suffix} found in {args.input_dir}.")
    
    # open one img to check for the size to determine the needed diffusion model
    test_img = utils.load_image(os.path.join(args.input_dir, target_files[0]), device_str)
    if test_img.shape == torch.Size([1, 3, 512, 512]):
        model_id = "stabilityai/stable-diffusion-2-1-base"
    elif test_img.shape == torch.Size([1, 3, 768, 768]):
        model_id = "stabilityai/stable-diffusion-2-1"
    else:
        raise ValueError(f"Invalid input shape: {test_img.shape}. Expected [1, 3, 512, 512] or [1, 3, 768, 768].")

    turbo_ddcm = TurboDDCM(model_id, args.T, args.K, args.M, args.seed, args.float32, device_str)
    runtimes = []
    for file_name in tqdm(target_files):
        img = utils.load_image(os.path.join(args.input_dir, file_name), device_str)
        weight_pixel_vector = None
        if args.weights_dir is not None:
            weight_pixel_vector = torch.load(os.path.join(args.weights_dir, os.path.splitext(file_name)[0] + '.pt'), device_str)

        compr_start_time = time.process_time()
        (reconstruction, encoding), compr_end_time = turbo_ddcm.compress(img, weight_pixel_vector)
        
        binary_path = os.path.join(args.output_dir, os.path.splitext(file_name)[0] + utils.BIN_SUFFIX)
        utils.save_as_binary(encoding, binary_path)
        runtimes.append({'file_name' : file_name, 'compression_time_seconds' : compr_end_time-compr_start_time})
        if args.save_reconstructions:
            utils.save_decoded_img(os.path.join(args.output_dir, file_name), reconstruction)

    if args.save_runtimes:
        pd.DataFrame(runtimes).to_csv(os.path.join(args.output_dir, 'compression_runtimes.csv'))

    args.model_id = model_id
    with open(os.path.join(args.output_dir, 'compression_config.json'), 'w') as f: # save config for decompression
        json.dump(vars(args), f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images to compress')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--M', type=int, required=True, help='Atoms to be chosen from the codebook in each diffusion step')
    
    # optional
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--float32', action='store_true', help='Use float32 precision for model inference')
    parser.add_argument('--seed', type=int, default=88888888, help='Random seed')
    
    parser.add_argument('--T', type=int, default=20, help='Compress using T diffusion steps')
    parser.add_argument('--K', type=int, default=16384, help="Codebook size")

    parser.add_argument('--weights_dir', type=str, default=None, help='Directory with priority maps')

    parser.add_argument('--save_reconstructions', action='store_true', default=False, help='Save reconstructions using the compression process')
    parser.add_argument('--save_runtimes', action='store_true', default=False, help='Save compression times in csv file')
    
    args = parser.parse_args()

    main(args)
