import torch
from torch import nn
from models.encoders import backbone_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt


class E2Style(nn.Module):

	def __init__(self, opts):
		super(E2Style, self).__init__()
		self.set_opts(opts)
		self.stage = self.opts.training_stage if self.opts.is_training is True else self.opts.stage
		self.encoder_firststage = backbone_encoders.BackboneEncoderFirstStage(50, 'ir_se', self.opts)

		if self.stage > 1:
			self.encoder_refinestage_list = nn.ModuleList([backbone_encoders.BackboneEncoderRefineStage(50, 'ir_se', self.opts) for i in range(self.stage-1)])

		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()


	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):
			if self.stage > self.opts.training_stage:
				raise ValueError(f'The stage must be no greater than {self.opts.training_stage} when testing!')
			print(f'Inference: Results are from Stage{self.stage}.', flush=True)
			print('Loading E2Style from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is not None) and self.opts.is_training:
			print(f'Train: The {self.stage}-th encoder of E2Style is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 2:
				for i in range(self.stage-2):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			print(f'Loading the {self.stage}-th encoder weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder_refinestage_list[self.stage-2].load_state_dict(encoder_ckpt, strict=False)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is None) and (self.stage==1) and self.opts.is_training:
			print(f'Train: The 1-th encoder of E2Style is to be trained.', flush=True)
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder_firststage.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=18)		


	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):

		stage_output_list = []
		if input_code:
			codes = x
		else:
			codes = self.encoder_firststage(x)
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else: 
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		input_is_latent = not input_code
		first_stage_output, result_latent = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
		stage_output_list.append(first_stage_output)

		if self.stage > 1:
			for i in range(self.stage-1):
				codes = codes + self.encoder_refinestage_list[i](x, self.face_pool(stage_output_list[i]))
				refine_stage_output, result_latent = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
				stage_output_list.append(refine_stage_output)

		if resize: 
			images = self.face_pool(stage_output_list[-1])

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None): 
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
