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
from utils.application_utils import move_latent

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

	# Load image, codes, and boundary.
	print(f'Loading images and corresponding inverted latent codes.')
	assert os.path.exists(f'{opts.edited_dir}/image_list.txt')
	assert os.path.exists(f'{opts.edited_dir}/latent_code.npy')
	image_list = []
	with open(os.path.join(opts.edited_dir, 'image_list.txt'), 'r') as f:
		for line in f:
			line = line.strip()
			image_list.append(line)
	latent_codes = np.load(os.path.join(opts.edited_dir, 'latent_code.npy'))
	assert latent_codes.shape[0] == len(image_list)
	num_images = latent_codes.shape[0]

	# Manipulate images.
	print(f'Start manipulation.')
	coeffs = [-40., -30., -20., -10., -5., -0., 10., 20., 30., 40.]
	step = len(coeffs)
	visualizer = HtmlPageVisualizer(num_rows=num_images, num_cols=step + 2, viz_size=256)
	visualizer.set_headers(['Name', 'Origin'] + [f'Step {i:02d}' for i in range(1, step + 1)])
	direction_file = opts.deriction_name+'.npy'
	assert os.path.exists(f'editing_directions/{direction_file}')
	
	for img_idx, img_name in enumerate(image_list):
		ori_image = resize_image(load_image(img_name), (256, 256))
		visualizer.set_cell(img_idx, 0, text=os.path.splitext(os.path.basename(img_name.strip()))[0])
		visualizer.set_cell(img_idx, 1, image=ori_image)
	for img_idx in tqdm(range(num_images), leave=False):
		output_images = move_latent(latent_codes[img_idx][np.newaxis], 'editing_directions/'+direction_file, coeffs, decoder)
		for s, output_image in enumerate(output_images):
			visualizer.set_cell(img_idx, s + 2, image=np.array(output_image))

	# Save results.
	visualizer.save(f'{opts.exp_dir}/{opts.deriction_name}.html')


if __name__ == '__main__':
	main()
