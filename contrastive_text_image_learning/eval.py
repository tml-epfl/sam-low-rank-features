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
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
import sys
sys.path.insert(0, 'vision_transformer')  # so that the local version of vit_jax is directly available

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import jax.numpy as jnp
import jax
import argparse
import tensorflow as tf
from flax import linen as nn
from flax.training import checkpoints as flax_checkpoints
# from flax.training.common_utils import shard, shard_prng_key
from transformers import FlaxBertModel, AutoConfig, AutoTokenizer
from models import ResNet18, get_image_model
from data import IMAGE_SIZE, get_coco_dataset_iter
from utils import calc_test_metrics_posthoc, compute_weight_matrix_rank


tf.get_logger().setLevel('INFO')
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('--bs', type=int, default=128, help='Batch size (128 = 10 GB is used)')
  parser.add_argument('--n_eval_batches', type=int, default=10, help='N batches')
  parser.add_argument('--projection_dim', type=int, default=768, help='Output feature dimension')
  parser.add_argument('--bottleneck_dim', type=int, default=-1, help='Linear bottleneck dimension')
  parser.add_argument('--model_path', type=str, default='', help='Model path')
  parser.add_argument('--split', type=str, default='test', help='Model path')
  parser.add_argument('--return_layer', type=int, default=-1, help='Which layer to return for GeLU and MLP output')
  args = parser.parse_args()


  max_text_length = 32
  text_model_checkpoint = 'bert-base-cased'

  # import ipdb;ipdb.set_trace()
  text_config = AutoConfig.from_pretrained(text_model_checkpoint)
  text_config.attention_probs_dropout_prob = 0
  text_config.hidden_dropout_prob = 0 
  text_config._name_or_path = None
  text_model = FlaxBertModel(config=text_config) 
  tokenizer = AutoTokenizer.from_pretrained(text_model_checkpoint)

  if args.bottleneck_dim > 0:
    text_proj = nn.Sequential([nn.Dense(args.bottleneck_dim), nn.Dense(args.projection_dim)])
    image_proj = nn.Sequential([nn.Dense(args.bottleneck_dim), nn.Dense(args.projection_dim)])
  else:
    text_proj = nn.Dense(args.projection_dim)
    image_proj = nn.Dense(args.projection_dim)

  rng = jax.random.PRNGKey(0)
  image_model_rng, rng = jax.random.split(rng)
  text_proj_rng, rng = jax.random.split(rng)

  image_model, image_params = get_image_model(random_init_image=False)
  image_state = None

  image_proj_params = image_proj.init(image_model_rng, jnp.ones((image_model.hidden_size, ))) 
  text_proj_params = text_proj.init(text_proj_rng, jnp.ones((text_config.hidden_size,)))

  models = [image_model, text_model, image_proj, text_proj]
  # params = [image_params, text_model.params, image_proj_params, text_proj_params]

  params_ckpt = flax_checkpoints.restore_checkpoint(args.model_path, target=None)['0']
  params = [params_ckpt['0'], params_ckpt['1'], params_ckpt['2'], params_ckpt['3']]
  
  # TODO: extract features from the BERT encoder as well (in a similar way; need a hacked HF repo here?)
  metrics = {}
  metrics = calc_test_metrics_posthoc(
      models,
      params,
      image_state,
      [rng, rng],
      tokenizer,
      max_text_length,
      args.bs,
      args.n_eval_batches,
      get_coco_dataset_iter(split=args.split, shuffle=False),
      args.split,
      args.return_layer
  )
  metrics['weight_matrix_ranks'] = compute_weight_matrix_rank(params, pc_threshold=0.99)
  metrics['rho'] = args.model_path.split('rho=')[1].split('_')[0]
  metrics['bottleneck_dim'] = args.bottleneck_dim
  metrics['split'] = args.split  # train or test
  metrics['return_layer'] = args.return_layer
  
  print('{},'.format(metrics))
 
