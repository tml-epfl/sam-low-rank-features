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

import sys
sys.path.insert(0, 'vision_transformer')

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # uncomment for dynamic memory allocation

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import jax.numpy as jnp
import jax
import argparse
import collections
import torch
import tensorflow as tf
import optax
import numpy as np
import tqdm
import flax
from flax import linen as nn
from flax.training import checkpoints as flax_checkpoints
# from flax.training.common_utils import shard, shard_prng_key
from transformers import FlaxBertModel, AutoConfig, AutoTokenizer
from vit_jax import input_pipeline
from models import ResNet18, get_image_model
from data import IMAGE_SIZE, get_coco_dataset_iter, get_batch_iter
from utils import calc_test_metrics, get_loss_fn, get_gradnorm_reg, sam_loss_grad_fn


def run(
    learning_rate=0.001,
    sam_rho=0.0,
    grad_norm_rho=0.0,
    seed=0,
    model_checkpoint='bert-base-cased',
    num_train_epochs=1,
    optimizer_name='adam',
    do_gradnorm_squared=False,
    total_batch_size=64,
    max_text_length=32,
    n_test_batches=1,
    temperature=1.0,
    projection_dim=768,
    bottleneck_dim=-1,
    export_dir='.',
    run_id='',
    random_init_image=False,
    random_init_text=False,
):
  tb_dir = os.path.join(export_dir, 'tb', run_id)
  if not tf.io.gfile.isdir(tb_dir):
    tf.io.gfile.makedirs(tb_dir)
  tb_writer = tf.summary.create_file_writer(tb_dir)

  text_config = AutoConfig.from_pretrained(model_checkpoint)
  text_config.attention_probs_dropout_prob = 0
  text_config.hidden_dropout_prob = 0 
  if random_init_text:
    text_model = FlaxBertModel(config=text_config)
  else:
    text_model = FlaxBertModel.from_pretrained(model_checkpoint, config=text_config)
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

  if bottleneck_dim > 0:
    text_proj = nn.Sequential([nn.Dense(bottleneck_dim), nn.Dense(projection_dim)])
    image_proj = nn.Sequential([nn.Dense(bottleneck_dim), nn.Dense(projection_dim)])
  else:
    text_proj = nn.Dense(projection_dim)
    image_proj = nn.Dense(projection_dim)

  rng = jax.random.PRNGKey(seed)
  image_model_rng, rng = jax.random.split(rng)
  text_proj_rng, rng = jax.random.split(rng)

  image_model, image_params = get_image_model(random_init_image)
  image_state = None

  # dropout_rngs = jax.random.split(rng, jax.local_device_count())
  text_proj_params = text_proj.init(text_proj_rng, jnp.ones((text_config.hidden_size,)))
  image_proj_params = image_proj.init(image_model_rng, jnp.ones((image_model.hidden_size, ))) 

  models = [image_model, text_model, image_proj, text_proj]
  params = [image_params, text_model.params, image_proj_params, text_proj_params]

  num_train_samples = 82783
  n_iters_per_epoch = num_train_samples // total_batch_size
  n_total_iters = num_train_epochs * n_iters_per_epoch

  lr_scheduler = optax.cosine_decay_schedule(learning_rate, decay_steps=n_total_iters, alpha=0.0)
  if optimizer_name == 'adam':
    tx = optax.adam(learning_rate=lr_scheduler)
  elif optimizer_name == 'sgd':
    tx = optax.sgd(learning_rate=lr_scheduler)
  else:
    assert False

  opt_state = tx.init(params)
  loss_fn = get_loss_fn(models)

  assert not (sam_rho > 0 and grad_norm_rho > 0)
  if sam_rho > 0.0:
    loss_grad_fn = sam_loss_grad_fn(loss_fn, sam_rho=sam_rho, has_aux=True)
  elif grad_norm_rho > 0.0:
    loss_grad_fn = jax.value_and_grad(
        get_gradnorm_reg(
            loss_fn,
            rho=grad_norm_rho,
            do_gradnorm_squared=do_gradnorm_squared,
            has_aux=True,
        ),
        has_aux=True,
    )
  else:
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  def train_step(params, batch, image_state, opt_state, dropout_rng):
    left_dropout_rng, right_dropout_rng, new_dropout_rng = jax.random.split(
        dropout_rng, 3
    )

    (loss_val, image_state), grads = loss_grad_fn(
        params,
        batch,
        image_state,
        dropout_rngs=[left_dropout_rng, right_dropout_rng],
        train=True,
        T=temperature,
    )
    # grads = jax.lax.pmean(grads, "batch")

    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # metrics = jax.lax.pmean({"loss": loss_val}, axis_name="batch")
    metrics = {'loss': loss_val}
    return params, new_dropout_rng, image_state, metrics

  # per_device_batch_size = 4
  # total_batch_size = per_device_batch_size * jax.local_device_count()

  rand = np.random.RandomState(seed)

  # training loop
  metrics = collections.defaultdict(list)
  step = 0

  for _, epoch in enumerate(
      tqdm.tqdm(
          range(1, num_train_epochs + 1),
          desc=f'Epoch ...',
          position=0,
          leave=True,
      )
  ):
    # train
    with tqdm.tqdm(
        total=n_iters_per_epoch,
        desc='Training...',
        leave=False,
    ) as progress_bar_train:
      ds = get_coco_dataset_iter(split='train', shuffle=True)
      for batch in get_batch_iter(
          ds=ds,
          tokenizer=tokenizer,
          max_text_length=max_text_length,
          total_batch_size=total_batch_size,
          rand=rand,
      ):  # TODO: get_batch_iter takes a few sec!
        params, rng, image_state, train_metrics = train_step(
            params, batch, image_state, opt_state, rng
        )  # TODO: first run takes a long time and only 1 CPU core is occupied at first
        # params, dropout_rngs, image_state = parallel_train_step(params, shard(batch), shard(image_state), opt_state, dropout_rngs)
        # train_loss_val = round(flax.jax_utils.unreplicate(train_metrics)['loss'].item(), 3)
        
        loss_val = train_metrics['loss'].item()
        metrics['train_loss'].append(loss_val)
        if tb_writer:
          with tb_writer.as_default():
            tf.summary.scalar('train_loss', loss_val, step=step)

        if step % 10 == 0:
          print('[step={}] train loss {:.3f}'.format(step, np.mean(metrics['train_loss'][-5:])))

        if step % 100 == 0:
          m_test = calc_test_metrics(
              models,
              params,
              image_state,
              [rng, rng],
              loss_fn,
              temperature,
              tokenizer,
              max_text_length,
              total_batch_size,
              n_test_batches,
              get_coco_dataset_iter(split='test', shuffle=False),
              'test',
          )
          print('[step={}] {}'.format(step, m_test))   
          m = calc_test_metrics(
              models,
              params,
              image_state,
              [rng, rng],
              loss_fn,
              temperature,
              tokenizer,
              max_text_length,
              total_batch_size,
              n_test_batches,
              get_coco_dataset_iter(split='train', shuffle=False),
              'train',
          )
          print('[step={}] {}'.format(step, m))   
          m.update(m_test)
          for k, v in m.items():
            metrics[k].append(v)
          if tb_writer:
            with tb_writer.as_default():
              for k, v in m.items():
                tf.summary.scalar(k, v, step=step)
        
        progress_bar_train.update(1)
        step += 1

      try:
        checkpoint_path = flax_checkpoints.save_checkpoint(
            f'{export_dir}/models/{run_id}', (params, opt_state, step), step, overwrite=True, keep=5)
        # params, opt_state, initial_step = flax_checkpoints.restore_checkpoint(
        #   workdir, (params, opt_state, initial_step))
        print('Saved the model at {}'.format(checkpoint_path))
      except:
        print('Failed to save the model.')

  for k, v in metrics.items():
    metrics[k] = np.array(v)
  return metrics


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--rho', type=float, default=0.0, help='Radius of SAM')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--bs', type=int, default=32, help='Batch size')
parser.add_argument('--projection_dim', type=int, default=768, help='Output feature dimension')
parser.add_argument('--bottleneck_dim', type=int, default=-1, help='Output feature dimension')
parser.add_argument('--random_init_image', action='store_true', help='Use random init instead of a pretrained image model')
parser.add_argument('--random_init_text', action='store_true', help='Use random init instead of a pretrained text model')
parser.add_argument('--run_name', type=str, default='', help='Append this string to run_id')
args = parser.parse_args()

# used for Tensorboard and for saving the models
run_id = 'lr={}_rho={}_random_init_image={}_random_init_text={}_bottleneck_dim={}'.format(
  args.lr, args.rho, args.random_init_image, args.random_init_text, args.bottleneck_dim)
if args.run_name != '':
  run_id += '_' + args.run_name  

run(
    learning_rate=args.lr,
    sam_rho=args.rho,
    grad_norm_rho=0.,
    seed=0,
    model_checkpoint="bert-base-cased",
    num_train_epochs=args.epochs,
    optimizer_name='adam',
    do_gradnorm_squared=True,
    total_batch_size=args.bs,
    max_text_length=32,
    n_test_batches=10,
    temperature=0.05,
    projection_dim=args.projection_dim,
    bottleneck_dim=args.bottleneck_dim,
    export_dir='/mnt/main-disk',
    run_id=run_id,
    random_init_image=args.random_init_image,
    random_init_text=args.random_init_text,
)
