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
import matlab
import matlab.engine


def gen_image_from_code(given_z, decoder):
	with torch.no_grad():
		temp_images, _ = decoder([torch.from_numpy(given_z).float().cuda()],input_is_latent=True,randomize_noise=False,return_latents=False)
	return np.array(tensor2im(temp_images[0]))


def gen_stegoed_image(cover, message, lastbit0image, eng):
	stego = eng.onlyforembed(matlab.uint8(np.expand_dims(cover,1).tolist()),matlab.uint8(np.expand_dims(message,1).tolist()))
	stego_numpy = np.array(stego._data.tolist()).astype('int8')
	stegoed_image = stego_numpy + lastbit0image
	return stegoed_image

def extract_message(stego, message_lenth, eng):
	message = eng.stc_extract(matlab.uint8(np.expand_dims(stego,1).tolist()), message_lenth, 10)
	message_numpy = np.array(message._data.tolist()).astype('int8')
	return message_numpy

def encoding_latent_code(reshaped_code): 
	temp = np.zeros((9*reshaped_code.shape[0])).astype('int8')
	for i in range(reshaped_code.shape[0]):
		if reshaped_code[i]>=0:
			temp[9*i]=0
		else:
			temp[9*i]=1
		integer_part = bin(int(str(abs(reshaped_code[i])).split('.')[0]))[2:].zfill(8)
		for j, binary in enumerate(integer_part):
			temp[9*i+1+j] = binary
	return temp

def decoding_latent_code(encoded_code): 
	decode_temp = np.random.rand(int(encoded_code.shape[0]/9)).astype('float32')
	for i in range(decode_temp.shape[0]):
		encoded_binary = encoded_code[9*i:9*(i+1)]
		integer_part = ''
		for binary in encoded_binary[1::]:
			integer_part+=str(binary)

		decoded_str = ''
		if encoded_binary[0]==1:
			decoded_str+='-'
		decoded_str+=str(int(integer_part,2))
		decode_temp[i] = eval(decoded_str)
	return decode_temp

def get_cover_and_lastbit0image(reshaped_image):
	cover_temp = np.zeros((reshaped_image.shape[0])).astype('int8')
	for i in range(reshaped_image.shape[0]):
		last_bit = int(bin(reshaped_image[i])[2:].zfill(8)[-1])
		cover_temp[i] = last_bit
	lastbit0image = reshaped_image - cover_temp
	return cover_temp, lastbit0image


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

	src_dir = opts.secret_dir
	src_dir_name = os.path.basename(src_dir.rstrip('/'))
	assert os.path.exists(src_dir)
	assert os.path.exists(f'{src_dir}/image_list.txt')
	assert os.path.exists(f'{src_dir}/latent_code.npy')
	dst_dir = opts.cover_dir
	dst_dir_name = os.path.basename(dst_dir.rstrip('/'))
	assert os.path.exists(dst_dir)
	assert os.path.exists(f'{dst_dir}/image_list.txt')
	job_name = f'HIDE_{src_dir_name}_TO_{dst_dir_name}'

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

	num_dst = len(dst_list)

	image_size = 1024
	engine = matlab.engine.start_matlab()
	engine.addpath('matlabapi')

	print(f'Start Hiding.')
	# Initialize visualizer.
	headers = ['Secret Image', 'Inversion Image', 'Cover Image', 'Stego Image', 'Recovered Image', 'Stego-Cover', 'Recovered-Secret']
	visualizer = HtmlPageVisualizer(num_rows=num_src * num_dst, num_cols=len(headers), viz_size=256)
	visualizer.set_headers(headers)

	for src_idx in tqdm(range(num_src), leave=False):
		image = resize_image(load_image(src_list[src_idx]), (image_size, image_size))
		inv_image = resize_image(gen_image_from_code(src_codes[src_idx:src_idx + 1], decoder), (image_size, image_size))

		#Generate message
		prox_random_code = np.rint(src_codes[src_idx])
		reshaped_random_latent_code = prox_random_code.reshape((np.size(prox_random_code)))
		message = encoding_latent_code(reshaped_random_latent_code)

		for dst_idx in tqdm(range(num_dst), leave=False):
			cover_image = resize_image(load_image(dst_list[dst_idx]), (image_size, image_size))
			#Generate cover
			temp_cover_image = cover_image.copy()
			reshaped_cover_image = temp_cover_image.reshape((np.size(temp_cover_image)))
			cover, lastbit0image = get_cover_and_lastbit0image(reshaped_cover_image)
			#Generate stego image
			stegoed_image = gen_stegoed_image(cover, message, lastbit0image, engine)
			reshaped_stegoed_image = stegoed_image.reshape(temp_cover_image.shape)
			#extract message from stego image
			stego, _ = get_cover_and_lastbit0image(reshaped_stegoed_image.reshape((np.size(temp_cover_image))))
			recovered_message = extract_message(stego, message.shape[0], engine)
			#recover latent code and image
			recovered_reshaped_latent_code = decoding_latent_code(recovered_message)
			recovered_latent_code = recovered_reshaped_latent_code.reshape(prox_random_code.shape)
			right_latent_code = recovered_latent_code.astype(np.float32)
			recovered_image_from_code = gen_image_from_code(right_latent_code[np.newaxis], decoder)

			stego_cover = reshaped_stegoed_image / 255. - cover_image / 255.
			stego_cover = stego_cover*10 + 0.5
			stego_cover = np.clip(stego_cover, 0.0, 1.0)
			stego_cover = stego_cover*255

			recovered_secret = recovered_image_from_code / 255. - image / 255.
			recovered_secret = recovered_secret*10 + 0.5
			recovered_secret = np.clip(recovered_secret, 0.0, 1.0)
			recovered_secret = recovered_secret*255


			visualizer.set_cell(num_dst*src_idx+dst_idx, 0, image=image)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 1, image=inv_image)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 2, image=cover_image)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 3, image=reshaped_stegoed_image)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 4, image=recovered_image_from_code)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 5, image=stego_cover)
			visualizer.set_cell(num_dst*src_idx+dst_idx, 6, image=recovered_secret)


	# Save results.
	visualizer.save(f'{opts.exp_dir}/{job_name}.html')


if __name__ == '__main__':
	main()


