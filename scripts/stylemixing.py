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
from utils.application_utils import mix_style


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

	style_dir = opts.style_dir
	style_dir_name = os.path.basename(style_dir.rstrip('/'))
	assert os.path.exists(style_dir)
	assert os.path.exists(f'{style_dir}/image_list.txt')
	assert os.path.exists(f'{style_dir}/latent_code.npy')
	content_dir = opts.content_dir
	content_dir_name = os.path.basename(content_dir.rstrip('/'))
	assert os.path.exists(content_dir)
	assert os.path.exists(f'{content_dir}/image_list.txt')
	assert os.path.exists(f'{content_dir}/latent_code.npy')
	job_name = f'{style_dir_name}_STYLIZE_{content_dir_name}'
	mix_layers = list(range(opts.mix_layer_start_idx, 18))

	# Load image and codes.
	print(f'Loading images and corresponding inverted latent codes.')
	style_list = []
	with open(f'{style_dir}/image_list.txt', 'r') as f:
		for line in f:
			line = line.strip()
			style_list.append(line)
	print(f'Loading inverted latent codes.')
	style_codes = np.load(f'{style_dir}/latent_code.npy')
	assert style_codes.shape[0] == len(style_list)
	num_styles = style_codes.shape[0]
	content_list = []
	with open(f'{content_dir}/image_list.txt', 'r') as f:
		for line in f:
			line = line.strip()
			content_list.append(line)
	print(f'Loading inverted latent codes.')
	content_codes = np.load(f'{content_dir}/latent_code.npy')
	assert content_codes.shape[0] == len(content_list)
	num_contents = content_codes.shape[0]

	# Mix styles.
	print(f'Start style mixing.')
	visualizer = HtmlPageVisualizer(num_rows=num_styles + 1, num_cols=num_contents + 1, viz_size=256)
	visualizer.set_headers(['Style'] +[f'Content {i:03d}' for i in range(num_contents)])
	for style_idx, style_name in enumerate(style_list):
		style_image = resize_image(load_image(style_name), (256, 256))
		visualizer.set_cell(style_idx + 1, 0, image=style_image)
	for content_idx, content_name in enumerate(content_list):
		content_image = resize_image(load_image(content_name), (256, 256))
		visualizer.set_cell(0, content_idx + 1, image=content_image)

	codes = mix_style(style_codes=style_codes,content_codes=content_codes,num_layers=18,mix_layers=mix_layers)
	for style_idx in tqdm(range(num_styles), leave=False):
		with torch.no_grad():
			temp_images, _ = decoder([torch.from_numpy(codes[style_idx]).float().cuda()],input_is_latent=True,randomize_noise=False,return_latents=False)
			x_init_inv = torch.nn.AdaptiveAvgPool2d((256, 256))(temp_images)
		output_images = [tensor2im(x_init_inv[i]) for i in range(x_init_inv.shape[0])]

		for content_idx, output_image in enumerate(output_images):
			visualizer.set_cell(style_idx + 1, content_idx + 1, image=np.array(output_image))

	# Save results.
	visualizer.save(f'{opts.exp_dir}/{job_name}.html')


if __name__ == '__main__':
	main()
