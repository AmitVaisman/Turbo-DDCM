import argparse

from compress import main as compress_main
from decompress import main as decompress_main

def main(args):
    compression_args = args
    compression_args.output_dir = args.output_compression_dir
    compress_main(compression_args)

    decompression_args = args
    decompression_args.input_dir = args.output_compression_dir
    decompression_args.output_dir = args.output_decompression_dir
    decompress_main(decompression_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images to compress')
    parser.add_argument('--output_compression_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--output_decompression_dir', type=str, required=True, help='Directory to save the results')
    
    parser.add_argument('--M', type=int, required=True, help='Atoms to be chosen from the codebook in each diffusion step')
    
    # optional
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--float32', action='store_true', help='Use float32 precision for model inference')
    parser.add_argument('--seed', type=int, default=88888888, help='Random seed')
    
    parser.add_argument('--T', type=int, default=20, help='Compress using T diffusion steps')
    parser.add_argument('--K', type=int, default=16384, help="Codebook size")

    parser.add_argument('--weights_dir', type=str, default=None, help='Directory with priority maps')

    parser.add_argument('--save_reconstructions', action='store_true', default=False, help='Save reconstructions')
    parser.add_argument('--save_runtimes', action='store_true', default=False, help='Save compression times in csv file')

    args = parser.parse_args()
    
    main(args)