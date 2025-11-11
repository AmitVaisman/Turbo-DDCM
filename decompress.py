import os
import argparse
import torch
import pandas as pd
import time
from tqdm import tqdm
from argparse import Namespace
import json

from turbo_ddcm.turbo_ddcm import TurboDDCM
from turbo_ddcm import utils

def main(args):
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    files = sorted(os.listdir(args.input_dir))
    target_files = [f for f in files if f.endswith(utils.BIN_SUFFIX)]
    if not target_files:
        raise FileNotFoundError(f"No files with suffix {args.img_suffix} found in {args.input_dir}.")

    # open config
    with open(os.path.join(args.input_dir, 'compression_config.json'), 'r') as f:
        compression_config = json.load(f)
    compression_config = Namespace(**compression_config)
    
    turbo_ddcm = TurboDDCM(compression_config.model_id, compression_config.T, compression_config.K, compression_config.M, compression_config.seed, compression_config.float32, device_str)
    runtimes = []
    for file_name in tqdm(target_files):
        encoding = utils.load_binary(os.path.join(args.input_dir, file_name))
        
        decompr_start_time = time.process_time()
        reconstruction = turbo_ddcm.decompress(encoding)
        decompr_end_time = time.process_time()
        
        utils.save_decoded_img(os.path.join(args.output_dir, os.path.splitext(file_name)[0] + '.png'), reconstruction)
        runtimes.append({'file_name' : os.path.splitext(file_name)[0] + '.png', 'decompression_time_seconds' : decompr_end_time-decompr_start_time})

    if args.save_runtimes:
        decompression_runtime_df = pd.DataFrame(runtimes)
        
        # join to compression runtime file (if exists) and compute the roundtrip time
        if os.path.exists(os.path.join(args.input_dir, 'compression_runtimes.csv')):
            compression_runtime_df = pd.read_csv(os.path.join(args.input_dir, 'compression_runtimes.csv'))
            decompression_runtime_df = decompression_runtime_df.merge(compression_runtime_df, on='file_name')
            decompression_runtime_df['roundtrip_time_seconds'] = decompression_runtime_df['compression_time_seconds'] + decompression_runtime_df['decompression_time_seconds']
            decompression_runtime_df = decompression_runtime_df[['file_name', 'compression_time_seconds', 'decompression_time_seconds', 'roundtrip_time_seconds']]

        decompression_runtime_df.to_csv(os.path.join(args.output_dir, 'decompression_runtimes.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing compressed files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')
    
    # optional
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    
    parser.add_argument('--save_runtimes', action='store_true', default=False, help='Save decompression times in csv file')
    
    args = parser.parse_args()
    
    main(args)
