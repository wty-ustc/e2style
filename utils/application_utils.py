# python 3.7
"""Utility functions for image editing from latent space."""
import torch
import os.path
import numpy as np
from utils.common import tensor2im


def parse_indices(obj, min_val=None, max_val=None):
  """Parses indices.

  If the input is a list or tuple, this function has no effect.

  The input can also be a string, which is either a comma separated list of
  numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
  be ignored.

  Args:
    obj: The input object to parse indices from.
    min_val: If not `None`, this function will check that all indices are equal
      to or larger than this value. (default: None)
    max_val: If not `None`, this function will check that all indices are equal
      to or smaller than this field. (default: None)

  Returns:
    A list of integers.

  Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
  """
  if obj is None or obj == '':
    indices = []
  elif isinstance(obj, int):
    indices = [obj]
  elif isinstance(obj, (list, tuple, np.ndarray)):
    indices = list(obj)
  elif isinstance(obj, str):
    indices = []
    splits = obj.replace(' ', '').split(',')
    for split in splits:
      numbers = list(map(int, split.split('-')))
      if len(numbers) == 1:
        indices.append(numbers[0])
      elif len(numbers) == 2:
        indices.extend(list(range(numbers[0], numbers[1] + 1)))
  else:
    raise ValueError(f'Invalid type of input: {type(obj)}!')

  assert isinstance(indices, list)
  indices = sorted(list(set(indices)))
  for idx in indices:
    assert isinstance(idx, int)
    if min_val is not None:
      assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
    if max_val is not None:
      assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

  return indices


def interpolate(src_codes, dst_codes, step=5):
  """Interpolates two sets of latent codes linearly.

  Args:
    src_codes: Source codes, with shape [num, *code_shape].
    dst_codes: Target codes, with shape [num, *code_shape].
    step: Number of interplolation steps, with source and target included. For
      example, if `step = 5`, three more samples will be inserted. (default: 5)

  Returns:
    Interpolated codes, with shape [num, step, *code_shape].

  Raises:
    ValueError: If the input two sets of latent codes are with different shapes.
  """
  if not (src_codes.ndim >= 2 and src_codes.shape == dst_codes.shape):
    raise ValueError(f'Shapes of source codes and target codes should both be '
                     f'[num, *code_shape], but {src_codes.shape} and '
                     f'{dst_codes.shape} are received!')
  num = src_codes.shape[0]
  code_shape = src_codes.shape[1:]

  a = src_codes[:, np.newaxis]
  b = dst_codes[:, np.newaxis]
  l = np.linspace(0.0, 1.0, step).reshape(
      [step if axis == 1 else 1 for axis in range(a.ndim)])
  results = a + l * (b - a)
  assert results.shape == (num, step, *code_shape)

  return results


def mix_style(style_codes,
              content_codes,
              num_layers=1,
              mix_layers=None,
              is_style_layerwise=True,
              is_content_layerwise=True):
  """Mixes styles from style codes to those of content codes.

  Each style code or content code consists of `num_layers` codes, each of which
  is typically fed into a particular layer of the generator. This function mixes
  styles by partially replacing the codes of `content_codes` from some certain
  layers with those of `style_codes`.

  For example, if both style code and content code are with shape [10, 512],
  meaning to have 10 layers and each employs a 512-dimensional latent code. And
  the 1st, 2nd, and 3rd layers are the target layers to perform style mixing.
  Then the top half of the content code (with shape [3, 512]) will be replaced
  by the top half of the style code (also with shape [3, 512]).

  NOTE: This function also supports taking single-layer latent codes as inputs,
  i.e., setting `is_style_layerwise` or `is_content_layerwise` as False. In this
  case, the corresponding code will be first repeated for `num_layers` before
  performing style mixing.

  Args:
    style_codes: Style codes, with shape [num_styles, *code_shape] or
      [num_styles, num_layers, *code_shape].
    content_codes: Content codes, with shape [num_contents, *code_shape] or
      [num_contents, num_layers, *code_shape].
    num_layers: Total number of layers in the generative model. (default: 1)
    mix_layers: Indices of the layers to perform style mixing. `None` means to
      replace all layers, in which case the content code will be completely
      replaced by style code. (default: None)
    is_style_layerwise: Indicating whether the input `style_codes` are
      layer-wise codes. (default: True)
    is_content_layerwise: Indicating whether the input `content_codes` are
      layer-wise codes. (default: True)
    num_layers

  Returns:
    Codes after style mixing, with shape [num_styles, num_contents, num_layers,
      *code_shape].

  Raises:
    ValueError: If input `content_codes` or `style_codes` is with invalid shape.
  """
  if not is_style_layerwise:
    style_codes = style_codes[:, np.newaxis]
    style_codes = np.tile(
        style_codes,
        [num_layers if axis == 1 else 1 for axis in range(style_codes.ndim)])
  if not is_content_layerwise:
    content_codes = content_codes[:, np.newaxis]
    content_codes = np.tile(
        content_codes,
        [num_layers if axis == 1 else 1 for axis in range(content_codes.ndim)])

  if not (style_codes.ndim >= 3 and style_codes.shape[1] == num_layers and
          style_codes.shape[1:] == content_codes.shape[1:]):
    raise ValueError(f'Shapes of style codes and content codes should be '
                     f'[num_styles, num_layers, *code_shape] and '
                     f'[num_contents, num_layers, *code_shape] respectively, '
                     f'but {style_codes.shape} and {content_codes.shape} are '
                     f'received!')

  layer_indices = parse_indices(mix_layers, min_val=0, max_val=num_layers - 1)
  if not layer_indices:
    layer_indices = list(range(num_layers))

  num_styles = style_codes.shape[0]
  num_contents = content_codes.shape[0]
  code_shape = content_codes.shape[2:]

  s = style_codes[:, np.newaxis]
  s = np.tile(s, [num_contents if axis == 1 else 1 for axis in range(s.ndim)])
  c = content_codes[np.newaxis]
  c = np.tile(c, [num_styles if axis == 0 else 1 for axis in range(c.ndim)])

  from_style = np.zeros(s.shape, dtype=bool)
  from_style[:, :, layer_indices] = True
  results = np.where(from_style, s, c)
  assert results.shape == (num_styles, num_contents, num_layers, *code_shape)

  return results


def move_latent(latent_vector, direction_file, coeffs, decoder):
  direction = np.load(direction_file)
  output_images = []
  for i, coeff in enumerate(coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[0][:8] = (latent_vector[0] + coeff*direction)[:8]
    with torch.no_grad():
      temp_images, _ = decoder([torch.from_numpy(new_latent_vector).float().cuda()],input_is_latent=True,randomize_noise=False,return_latents=False)
      x_init_inv = torch.nn.AdaptiveAvgPool2d((256, 256))(temp_images)
    output_images.append(tensor2im(x_init_inv[0]))
  return output_images