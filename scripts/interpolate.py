import os
from argparse import Namespace
import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append(".")
sys.path.append("..")
from models.stylegan2.model import Generator
from models.e2style import get_keys
from utils.visualizer import load_image, resize_image
from utils.visualizer import HtmlPageVisualizer
from options.test_options import TestOptions
from utils.common import tensor2im
from utils.application_utils import interpolate


def main():
	"""Main function."""
	test_opts = TestOptions().parse()
	os.makedirs(test_opts.exp_dir, exist_ok=True)
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	# Load model.
	print(f'Loading generator.')
	decoder = Generator(1024, 512, 8)
	decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
	decoder.eval()
	decoder.cuda()

	src_dir = opts.source_dir
	src_dir_name = os.path.basename(src_dir.rstrip('/'))
	assert os.path.exists(src_dir)
	assert os.path.exists(f'{src_dir}/image_list.txt')
	assert os.path.exists(f'{src_dir}/latent_code.npy')
	dst_dir = opts.target_dir
	dst_dir_name = os.path.basename(dst_dir.rstrip('/'))
	assert os.path.exists(dst_dir)
	assert os.path.exists(f'{dst_dir}/image_list.txt')
	assert os.path.exists(f'{dst_dir}/latent_code.npy')
	job_name = f'{src_dir_name}_TO_{dst_dir_name}'

	# Load image and codes.
	print(f'Loading images and corresponding inverted latent codes.')
	src_list = []
	with open(f'{src_dir}/image_list.txt', 'r') as f:
		for line in f:
			line = line.strip()
			src_list.append(line)
	src_codes = np.load(f'{src_dir}/latent_code.npy')
	assert src_codes.shape[0] == len(src_list)
	num_src = src_codes.shape[0]
	dst_list = []
	with open(f'{dst_dir}/image_list.txt', 'r') as f:
		for line in f:
			line = line.strip()
			dst_list.append(line)
	dst_codes = np.load(f'{dst_dir}/latent_code.npy')
	assert dst_codes.shape[0] == len(dst_list)
	num_dst = dst_codes.shape[0]

	# Interpolate images.
	print(f'Start interpolation.')
	step = opts.step + 2
	visualizer = HtmlPageVisualizer(num_rows=num_src * num_dst, num_cols=step + 2, viz_size=256)
	visualizer.set_headers(['Source', 'Source Inversion'] +[f'Step {i:02d}' for i in range(1, step - 1)] +['Target Inversion', 'Target'])

	for src_idx in tqdm(range(num_src), leave=False):
		src_code = src_codes[src_idx:src_idx + 1]
		codes = interpolate(src_codes=np.repeat(src_code, num_dst, axis=0),dst_codes=dst_codes,step=step)
		for dst_idx in tqdm(range(num_dst), leave=False):
			with torch.no_grad():
				temp_images, _ = decoder([torch.from_numpy(codes[dst_idx]).float().cuda()],input_is_latent=True,randomize_noise=False,return_latents=False)
				x_init_inv = torch.nn.AdaptiveAvgPool2d((256, 256))(temp_images)
			output_images = [tensor2im(x_init_inv[i]) for i in range(x_init_inv.shape[0])]
			row_idx = src_idx * num_dst + dst_idx
			visualizer.set_cell(row_idx, 0, image=resize_image(load_image(src_list[src_idx]), (256, 256)))
			visualizer.set_cell(row_idx, step + 1, image=resize_image(load_image(dst_list[dst_idx]), (256, 256)))
			for s, output_image in enumerate(output_images):
				visualizer.set_cell(row_idx, s + 1, image=np.array(output_image))

	# Save results.
	visualizer.save(f'{opts.exp_dir}/{job_name}.html')


if __name__ == '__main__':
	main()
