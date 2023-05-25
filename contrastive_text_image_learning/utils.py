import numpy as np
import jax.numpy as jnp
import jax
import functools
from data import get_batch_iter


def gelu(x):
    """
    GeLU in numpy.
    """
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
    return x * cdf


def get_embeddings(models, params, batch, image_state, dropout_rngs, train, return_pre_head=False):
  del image_state
  # (batch_size, k) for both inputs
  # x_left, image_state = models[0].apply(
  #     {'params': params[0], **image_state},
  #     batch['image'],
  #     rngs={'dropout': dropout_rngs[0]},
  #     mutable=['batch_stats'],
  #     train=train,
  # )
  x_left_prehead = models[0].apply(
      {'params': params[0]},
      batch['image'],
      train=train,
  )
  x_right_prehead = models[1](
      input_ids=batch['input_ids'],
      attention_mask=batch['attention_mask'],
      token_type_ids=batch['token_type_ids'],
      train=train,
      params=params[1],  # note: using models[1].params freezes the text encoder!
      dropout_rng=dropout_rngs[1],
  )[0]

  x_left = models[2].apply(params[2], x_left_prehead)  # image proj
  x_right_prehead = jnp.mean(x_right_prehead, axis=1)  # mean pool across token positions
  x_right = models[3].apply(params[3], x_right_prehead)  # text proj

  x_left /= jnp.linalg.norm(x_left, ord=2, axis=1, keepdims=True)
  x_right /= jnp.linalg.norm(x_right, ord=2, axis=1, keepdims=True)

  if not return_pre_head:
    return x_left, x_right
  else:
    return x_left_prehead, x_left, x_right_prehead, x_right


def get_loss_fn(models):
  @functools.partial(jax.jit, static_argnames=['T', 'train', 'mask'])
  def loss_fn(
      params, batch, image_state, dropout_rngs, T=1, mask=0, train=True
  ):
    x_left, x_right = get_embeddings(
        models, params, batch, image_state, dropout_rngs, train
    )
    logits = mask + jnp.matmul(x_left, jnp.transpose(x_right)) / T
    right_ent = -jnp.log(1e-20 + jnp.diag(jax.nn.softmax(logits, axis=1)))
    left_ent = -jnp.log(1e-20 + jnp.diag(jax.nn.softmax(logits, axis=0)))

    loss_val = jnp.mean((right_ent + left_ent) / 2.0)
    return loss_val, image_state

  return loss_fn


def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t.

  ||x||_2 <= 1.
  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(
      sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)])
  )
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient


def get_gradnorm_reg(loss, rho, do_gradnorm_squared, has_aux=True):
  @functools.partial(jax.jit, static_argnames=['T', 'train', 'mask'])
  def new_loss_fn(
      params, batch, image_state, dropout_rngs, T=1, mask=0, train=True
  ):
    g = jax.grad(loss, has_aux=has_aux)(
        params, batch, image_state, dropout_rngs, T=T, mask=mask, train=train
    )
    if has_aux:
      g = g[0]
    g_norm_sq = sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g)]
    )
    reg = g_norm_sq if do_gradnorm_squared else jnp.sqrt(g_norm_sq)
    if has_aux:
      loss_val, aux_val = loss(
          params, batch, image_state, dropout_rngs, T=T, mask=mask, train=train
      )
      return loss_val + rho * reg, aux_val
    else:
      return (
          loss(
              params,
              batch,
              image_state,
              dropout_rngs,
              T=T,
              mask=mask,
              train=train,
          )
          + rho * reg
      )

  return new_loss_fn


def sam_loss_grad_fn(loss_fn, sam_rho=0.05, has_aux=True):
  loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)

  @functools.partial(jax.jit, static_argnames=['T', 'train', 'mask'])
  def fn(params, batch, image_state, dropout_rngs, T=1, mask=0, train=True):
    # params must be first arg
    if has_aux:
      (loss_val, aux_val), grad = loss_grad_fn(
          params, batch, image_state, dropout_rngs, T=T, mask=mask, train=train
      )
    else:
      loss_val, grad = loss_grad_fn(
          params, batch, image_state, dropout_rngs, T=T, mask=mask, train=train
      )

    grad = dual_vector(grad)
    adv_params = jax.tree_map(lambda a, b: a + sam_rho * b, params, grad)
    grad = jax.grad(loss_fn, has_aux=has_aux)(
        adv_params,
        batch,
        image_state,
        dropout_rngs,
        T=T,
        mask=mask,
        train=train,
    )
    if has_aux:
      grad = grad[0]
    grad = jax.tree_map(lambda x: x.astype(jnp.float32), grad)
    if has_aux:
      return (loss_val, aux_val), grad
    else:
      return loss_val, grad

  return fn


def calc_pca_rank(X, thresholds):
  # thresholds: [0.99, 0.95, 0.90, 0.85]
  # (n_examples, n_features)
  try:
    n_examples = X.shape[0]
    X = X - np.mean(X, axis=1, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    p_eigs = s**2 / (n_examples - 1)
    p_cumsum = np.cumsum(p_eigs)
    n_components = [
        np.sum(p_cumsum <= np.sum(p_eigs) * threshold) + 1
        for threshold in thresholds
    ]
    return n_components
  except Exception as _:
    return [-1] * len(thresholds)


def calc_test_metrics(
    models,
    params,
    image_state,
    dropout_rngs,
    loss_fn,
    temperature,
    tokenizer,
    max_text_length,
    total_batch_size,
    n_test_batches,
    ds,
    prefix,
):
  # image_state doesnt matter nor does dropout_rngs
  metrics = {}
  loss_val = []
  imagewise_accs = []
  textwise_accs = []
  x_left = []
  x_right = []
  for _, batch in zip(
      range(n_test_batches),
      get_batch_iter(
          ds=ds,
          tokenizer=tokenizer,
          max_text_length=max_text_length,
          total_batch_size=total_batch_size,
          rand=None,
      ),
  ):
    lv, _ = loss_fn(
        params, batch, image_state, dropout_rngs, train=False, T=temperature
    )
    loss_val.append(lv.item())
    xl, xr = get_embeddings(
        models, params, batch, image_state, dropout_rngs, False
    )
    logits = np.asarray(jnp.matmul(xl, jnp.transpose(xr)))
    bs = logits.shape[0]
    # logits is (batch_size, batch_size) where rows are images, columns are texts
    textwise_accs.append(100*np.mean(np.argmax(logits, axis=1) == np.arange(bs)))
    imagewise_accs.append(100*np.mean(np.argmax(logits, axis=0) == np.arange(bs)))
    x_left.append(np.asarray(xl))
    x_right.append(np.asarray(xr))

  thresholds = [0.99, 0.95, 0.90, 0.85]
  threshold_names = [99, 95, 90, 85]
  image_ranks = calc_pca_rank(np.concatenate(x_left, axis=0), thresholds)
  text_ranks = calc_pca_rank(np.concatenate(x_right, axis=0), thresholds)
  for i, name in enumerate(threshold_names):
    metrics[f'{prefix}_image_rank_{name}p'] = image_ranks[i]
    metrics[f'{prefix}_text_rank_{name}p'] = text_ranks[i]

  metrics[f'{prefix}_loss'] = np.mean(loss_val)
  metrics[f'{prefix}_imagewise_acc'] = np.mean(imagewise_accs)
  metrics[f'{prefix}_textwise_acc'] = np.mean(textwise_accs)
  return metrics


def compute_n_highly_corr_acts(acts, corr_threshold):
  corr_matrix = np.corrcoef(acts.T) 
  corr_matrix -= np.eye(corr_matrix.shape[0])
  idx_to_delete, i, j = [], 0, 0
  while i != corr_matrix.shape[0]:
      # print(i, corr_matrix.shape, (np.abs(corr_matrix[i]) > corr_threshold).sum())
      if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
          corr_matrix = np.delete(corr_matrix, (i), axis=0)
          corr_matrix = np.delete(corr_matrix, (i), axis=1)
          # print('delete', j)
          idx_to_delete.append(j)
      else:
          i += 1
      j += 1
  assert corr_matrix.shape[0] == corr_matrix.shape[1]
  return len(idx_to_delete)


def compute_weight_matrix_rank(params, pc_threshold=0.99):
  # params[0]['Transformer']['encoderblock_3']['MlpBlock_0']
  weight_matrices = [
    params[0]['Transformer']['encoderblock_3']['MlpBlock_0']['Dense_0']['kernel'],
    params[0]['Transformer']['encoderblock_3']['MlpBlock_0']['Dense_1']['kernel'],
    params[0]['Transformer']['encoderblock_5']['MlpBlock_0']['Dense_0']['kernel'],
    params[0]['Transformer']['encoderblock_5']['MlpBlock_0']['Dense_1']['kernel'],
    params[0]['Transformer']['encoderblock_7']['MlpBlock_0']['Dense_0']['kernel'],
    params[0]['Transformer']['encoderblock_7']['MlpBlock_0']['Dense_1']['kernel'],
    params[0]['Transformer']['encoderblock_9']['MlpBlock_0']['Dense_0']['kernel'],
    params[0]['Transformer']['encoderblock_9']['MlpBlock_0']['Dense_1']['kernel'],
  ]
  weight_matrix_ranks = []
  for weight_matrix in weight_matrices:
    weight_matrix_ranks.append(calc_pca_rank(weight_matrix, [pc_threshold])[0])
  return weight_matrix_ranks


def calc_test_metrics_posthoc(
    models,
    params,
    image_state,
    dropout_rngs,
    tokenizer,
    max_text_length,
    total_batch_size,
    n_test_batches,
    ds,
    prefix,
    return_layer,
):
  # image_state doesnt matter nor does dropout_rngs
  pc_threshold = 0.99
  corr_threshold = 0.95

  metrics = {}
  imagewise_accs, textwise_accs = [], []
  preatt_left, preacts_left, mlp_left, res_left = [], [], [], []
  x_left_prehead, x_left, x_right_prehead, x_right = [], [], [], []
  for _, batch in zip(
      range(n_test_batches),
      get_batch_iter(
          ds=ds,
          tokenizer=tokenizer,
          max_text_length=max_text_length,
          total_batch_size=total_batch_size,
          rand=None,
      ),
  ):
    xl_preatt, xl_preact, xl_mlp, xl_res = models[0].apply({'params': params[0]}, batch['image'], 
                                     train=False, return_acts=True, 
                                     return_layer=return_layer
    )

    xl_prehead, xl, xr_prehead, xr = get_embeddings(
        models, params, batch, image_state, dropout_rngs, False, return_pre_head=True
    )
    logits = np.asarray(jnp.matmul(xl, jnp.transpose(xr)))
    bs = logits.shape[0]
    # logits is (batch_size, batch_size) where rows are images, columns are texts
    textwise_accs.append(100*np.mean(np.argmax(logits, axis=1) == np.arange(bs)))
    imagewise_accs.append(100*np.mean(np.argmax(logits, axis=0) == np.arange(bs)))
    preatt_left.append(np.asarray(xl_preatt))
    preacts_left.append(np.asarray(xl_preact))
    mlp_left.append(np.asarray(xl_mlp))
    res_left.append(np.asarray(xl_res))
    x_left_prehead.append(np.asarray(xl_prehead))
    x_right_prehead.append(np.asarray(xr_prehead))
    x_left.append(np.asarray(xl))
    x_right.append(np.asarray(xr))
  
  x_left = np.concatenate(x_left, axis=0)
  x_right = np.concatenate(x_right, axis=0)
  x_left_prehead = np.concatenate(x_left_prehead, axis=0)
  x_right_prehead = np.concatenate(x_right_prehead, axis=0)
  preatt_left = np.concatenate(preatt_left, axis=0)
  preacts_left = np.concatenate(preacts_left, axis=0)
  mlp_left = np.concatenate(mlp_left, axis=0)
  res_left = np.concatenate(res_left, axis=0)

  token_embedding_to_take = 0
  if return_layer != 0:
    preatt_left = preatt_left[:, token_embedding_to_take, :]
    preacts_left = preacts_left[:, token_embedding_to_take, :]
    mlp_left = mlp_left[:, token_embedding_to_take, :]
    res_left = res_left[:, token_embedding_to_take, :]
  else:
    preatt_left = preatt_left[:, token_embedding_to_take, :].reshape(preatt_left.shape[0], -1)
    preacts_left = preacts_left[:, token_embedding_to_take, :].reshape(preacts_left.shape[0], -1)
    mlp_left = mlp_left[:, token_embedding_to_take, :].reshape(mlp_left.shape[0], -1)
    res_left = res_left[:, token_embedding_to_take, :].reshape(res_left.shape[0], -1)

  acts_left = gelu(preacts_left)
  premlp_left = res_left - mlp_left
  att_left = premlp_left - preatt_left

  # act_threshold = 0.2
  act_threshold = acts_left.max() / 10
  metrics['avg_pos_image_mlp'] = (mlp_left > act_threshold).mean()
  metrics['avg_pos_image_acts'] = (acts_left > act_threshold).mean()
  metrics['n_image_acts_activated_more_than_on_10p_inputs'] = ((acts_left > act_threshold).sum(0) > (acts_left.shape[0] / 10)).mean()
  metrics['n_image_acts_activated_more_than_on_1p_inputs'] = ((acts_left > act_threshold).sum(0) > (acts_left.shape[0] / 100)).mean()
  metrics['n_image_acts_activated_more_than_on_0p_inputs'] = ((acts_left > act_threshold).sum(0) > 0).mean()

  if return_layer != 0:  # too slow for return_layer==0
    metrics['n_highly_corr_image_acts'] = compute_n_highly_corr_acts(acts_left, corr_threshold=corr_threshold)
    metrics['n_highly_corr_final_image_embeddings'] = compute_n_highly_corr_acts(x_left, corr_threshold=corr_threshold)

  pc_threshold_name = str(pc_threshold).replace('0.', '')
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_preatt'] = calc_pca_rank(preatt_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_att'] = calc_pca_rank(att_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_premlp'] = calc_pca_rank(premlp_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_pregelu'] = calc_pca_rank(preacts_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_gelu'] = calc_pca_rank(acts_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_mlp'] = calc_pca_rank(mlp_left, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p_res'] = calc_pca_rank(res_left, [pc_threshold])[0]

  metrics[f'{prefix}_image_prehead_rank_{pc_threshold_name}p'] = calc_pca_rank(x_left_prehead, [pc_threshold])[0]
  metrics[f'{prefix}_text_prehead_rank_{pc_threshold_name}p'] = calc_pca_rank(x_right_prehead, [pc_threshold])[0]
  metrics[f'{prefix}_image_rank_{pc_threshold_name}p'] = calc_pca_rank(x_left, [pc_threshold])[0]
  metrics[f'{prefix}_text_rank_{pc_threshold_name}p'] = calc_pca_rank(x_right, [pc_threshold])[0]

  metrics[f'{prefix}_imagewise_acc'] = np.mean(imagewise_accs)
  metrics[f'{prefix}_textwise_acc'] = np.mean(textwise_accs)

  return metrics

