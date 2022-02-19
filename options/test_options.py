from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to E2Style model checkpoint')
		self.parser.add_argument('--stage', default=1, type=int, help='Results of stage i')
		self.parser.add_argument('--is_training', default=False, type=bool, help='Training or testing')
		self.parser.add_argument('--data_path', type=str, default='gt_images', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
		self.parser.add_argument('--save_inverted_codes', action='store_true', help='Whether to save the inverted latent codes')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')


		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='Downsampling factor for super-res (should be a single value for inference).')
		# arguments for semantic manipulation
		self.parser.add_argument('--deriction_name', default=None, type=str, help='Edited semantic name')
		self.parser.add_argument('--edited_dir', default=None, type=str, help='Edited directory')
		# arguments for stylemixing
		self.parser.add_argument('--style_dir', default=None, type=str, help='Style directory')
		self.parser.add_argument('--content_dir', default=None, type=str, help='Content directory')
		self.parser.add_argument('--mix_layer_start_idx', type=int, default=10, help='Style mixing is performed from this layer to the last layer. (default: 10)')
		# arguments for interpolation
		self.parser.add_argument('--source_dir', default=None, type=str, help='Source directory')
		self.parser.add_argument('--target_dir', default=None, type=str, help='Target directory')
		self.parser.add_argument('--step', type=int, default=5, help='Number of steps for interpolation. (default: 5)')
		# arguments for secure deep hiding
		self.parser.add_argument('--secret_dir', default=None, type=str, help='Secret directory')
		self.parser.add_argument('--cover_dir', default=None, type=str, help='Cover directory')
		

	def parse(self):
		opts = self.parser.parse_args()
		return opts