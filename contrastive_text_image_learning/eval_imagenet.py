# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

import logging
import os
logging.getLogger('tensorflow').disabled = True
logging.disable(logging.WARNING)  # key line to disable the annoying warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
sys.path.insert(0, 'vision_transformer')  # so that the local version of vit_jax is directly available

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import jax.numpy as jnp
import jax
import argparse
import tensorflow as tf
import numpy as np
from flax import linen as nn
# from flax.training.common_utils import shard, shard_prng_key
from data import IMAGE_SIZE, get_coco_dataset_iter, get_batch_iter_images
from utils import calc_pca_rank, gelu
from vit_jax import checkpoint as vit_checkpoint
from vit_jax import models as vit_models
from vit_jax.configs import models as vit_models_config


tf.get_logger().setLevel('INFO')
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('--bs', type=int, default=128, help='Batch size (128 = 10 GB is used)')
  parser.add_argument('--n_eval_batches', type=int, default=10, help='N batches')
  parser.add_argument('--model_path', type=str, default='', help='Model path')
  parser.add_argument('--split', type=str, default='test', help='Split: train or test')
  parser.add_argument('--return_layer', type=int, default=-1, help='Which layer to return')
  parser.add_argument('--avg_tokens', action='store_true')
  args = parser.parse_args()

  config_name = args.model_path[:-4].replace('-224', '').split('/')[-1]  # e.g., ViT-B_16, ViT-L_32, etc
  model_config = vit_models_config.MODEL_CONFIGS[config_name]
  resolution = 224 if 'sam/' in args.model_path or '224' in args.model_path or 'Mixer' in args.model_path else 384  # seems like 224 (and not 384) since this is what's mentioned in Chen et al. (2021)
  
  if config_name.startswith('Mixer'):
    model = vit_models.MlpMixer(num_classes=None, **model_config)
  else:
    model = vit_models.VisionTransformer(num_classes=None, **model_config)

  params = vit_checkpoint.load(args.model_path)  
  
  metrics = {'n_eval_batches': args.n_eval_batches, 'model_path': args.model_path, 'return_layer': args.return_layer}
  pc_threshold = 0.99
  pc_threshold_name = str(pc_threshold).replace('0.', '')
  
  preatt_all, preact_all, mlp_all, res_all = [], [], [], []
  ds = get_imagenet_dataset_iter(split=args.split, image_size=(resolution, resolution), shuffle=False)
  for _, batch in zip(
    range(args.n_eval_batches), 
    get_batch_iter_images(ds, args.bs)
  ):
    preatt, preact, mlp, res = model.apply({'params': params}, 2*(batch['image'] - 0.5), train=False, return_acts=True, return_layer=args.return_layer)
    preatt_all.append(np.asarray(preatt[:, 0, :] if not args.avg_tokens else preatt.mean(1)))
    preact_all.append(np.asarray(preact[:, 0, :] if not args.avg_tokens else preact.mean(1)))
    mlp_all.append(np.asarray(mlp[:, 0, :] if not args.avg_tokens else mlp.mean(1)))
    res_all.append(np.asarray(res[:, 0, :] if not args.avg_tokens else res.mean(1)))
  
  preatt_all = np.concatenate(preatt_all, axis=0)
  preact_all = np.concatenate(preact_all, axis=0)
  mlp_all = np.concatenate(mlp_all, axis=0)
  res_all = np.concatenate(res_all, axis=0)
  acts_all = gelu(preact_all)
  premlp_all = res_all - mlp_all
  att_all = premlp_all - preatt_all

  metrics[f'image_rank_{pc_threshold_name}p_preatt'] = calc_pca_rank(preatt_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_preact'] = calc_pca_rank(preact_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_mlp'] = calc_pca_rank(mlp_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_res'] = calc_pca_rank(res_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_acts'] = calc_pca_rank(acts_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_premlp'] = calc_pca_rank(premlp_all, [pc_threshold])[0]
  metrics[f'image_rank_{pc_threshold_name}p_att'] = calc_pca_rank(att_all, [pc_threshold])[0]

  print('{},'.format(metrics))


