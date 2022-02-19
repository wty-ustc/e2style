import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.e2style import E2Style


def run():
	test_opts = TestOptions().parse()

	if test_opts.resize_factors is not None:
		assert len(test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
		                                'downsampling_{}'.format(test_opts.resize_factors))
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
										'downsampling_{}'.format(test_opts.resize_factors))
	else:
		out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
		out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)


	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	opts = Namespace(**opts)

	net = E2Style(opts)
	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
	                           transform=transforms_dict['transform_inference'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	global_i = 0
	global_time = []
	latent_list = []
	image_list_path = os.path.join(opts.exp_dir, 'image_list.txt')
	with open(image_list_path, 'w') as f:
		for input_batch in tqdm(dataloader):
			if global_i >= opts.n_images:
				break
			with torch.no_grad():
				input_cuda = input_batch.cuda().float()
				tic = time.time()
				result_batch, latent_batch = net(input_cuda, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
				latent_list.append(latent_batch)
				toc = time.time()
				global_time.append(toc - tic)

			for i in range(opts.test_batch_size):
				result = tensor2im(result_batch[i])
				im_path = dataset.paths[global_i]
				f.write(im_path+'\r\n')

				if opts.couple_outputs or global_i % 100 == 0:
					input_im = log_input_image(input_batch[i], opts)
					resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
					if opts.resize_factors is not None:
						# for super resolution, save the original, down-sampled, and output
						source = Image.open(im_path)
						res = np.concatenate([np.array(source.resize(resize_amount)),
											  np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
											  np.array(result.resize(resize_amount))], axis=1)
					else:
						# otherwise, save the original and output
						res = np.concatenate([np.array(input_im.resize(resize_amount)),
											  np.array(result.resize(resize_amount))], axis=1)
					Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

				im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
				Image.fromarray(np.array(result)).save(im_save_path)

				global_i += 1

	f.close()
	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	if opts.save_inverted_codes:
		np.save(os.path.join(opts.exp_dir, f"latent_code.npy"), torch.cat(latent_list, 0).detach().cpu().numpy())
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)
	f.close()


if __name__ == '__main__':
	run()
