from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import cv2
import random
import skimage
from skimage import img_as_ubyte
class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def preprocessing_for_restoration(self, from_im, to_im):
		from_im = np.array(from_im)
		from_im = cv2.resize(from_im, (256,256))
		to_im = np.array(to_im)
		if np.random.uniform(0, 1) < 0.5:
			from_im = cv2.flip(from_im, 1)
			to_im = cv2.flip(to_im, 1)
		return from_im, to_im

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.opts.dataset_type == 'ffhq_colorization':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			from_im=cv2.cvtColor(from_im, cv2.COLOR_BGR2GRAY)
			from_im = np.expand_dims(from_im, axis=2)
			from_im = np.concatenate((from_im, from_im, from_im), axis=-1)
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        

		elif self.opts.dataset_type == 'ffhq_denoise':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			if random.random()>0.5:
				from_im = skimage.util.random_noise(from_im, mode='gaussian', var=0.01)
			else:
				from_im = skimage.util.random_noise(from_im, mode='s&p')
			from_im = img_as_ubyte(from_im)
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        
		
		elif self.opts.dataset_type == 'ffhq_inpainting':
			from_im, to_im = self.preprocessing_for_restoration(from_im, to_im)
			a = [np.random.choice([35,220],1)[0], 35]
			b = [np.random.choice([35,70],1)[0], 220]
			c = [b[0]+150, 220]
			triangle = np.array([a, b, c])
			from_im = cv2.fillConvexPoly(from_im, triangle, (0, 0, 0))
			from_im = Image.fromarray(from_im.astype('uint8')).convert('RGB')
			to_im = Image.fromarray(to_im.astype('uint8')).convert('RGB')        
		
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im
