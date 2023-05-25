import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import numpy as np
import jax.numpy as jnp


IMAGE_SIZE = 384


def coco_image_processor(feature_dict, image_size):
  image = feature_dict['image']
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, image_size)
  image /= 255.0
  feature_dict['image'] = image
  return feature_dict


COCO_BUILDER = None
def get_coco_dataset_iter(split='train', image_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True):
  global COCO_BUILDER
  if COCO_BUILDER is None:
    COCO_BUILDER = tfds.builder('coco_captions')
    COCO_BUILDER.download_and_prepare()

  ds = COCO_BUILDER.as_dataset(split=split).map(
      partial(coco_image_processor, image_size=image_size)
  )
  if shuffle:
    ds = ds.shuffle(10000)

  return ds.as_numpy_iterator()


def get_batch_iter_images(ds, total_batch_size):
  while True:
    image_inputs = []
    for _, d in zip(range(total_batch_size), ds):
      image_inputs.append(d['image'])
    if len(image_inputs) != total_batch_size:
      break
    else:
      batch = {}
      batch['image'] = np.stack(image_inputs, axis=0)
      batch = {k: jnp.array(v) for k, v in batch.items()}
      yield batch


def get_batch_iter(ds, tokenizer, max_text_length, total_batch_size, rand):
  while True:
    text_inputs = []
    image_inputs = []
    for _, d in zip(range(total_batch_size), ds):
      texts = d['captions']['text']
      if rand:
        text = texts[rand.randint(len(texts))]
      else:
        text = texts[0]
      text_inputs.append(str(text))
      # image_inputs.append(np.swapaxes(d['image'], 0, 2))
      image_inputs.append(d['image'])
    if len(text_inputs) != total_batch_size:
      break
    else:
      batch = tokenizer(
          text_inputs,
          padding='max_length',
          max_length=max_text_length,
          truncation=True,
          return_tensors='np',
      )
      batch['image'] = np.stack(image_inputs, axis=0)
      batch = {k: jnp.array(v) for k, v in batch.items()}
      yield batch


def create_split(dataset_builder, batch_size, dtype=tf.float32,
                 image_size=IMAGE_SIZE, cache=False):
  """Creates a split from the dataset using TensorFlow Datasets.
  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  ds = ds.repeat()
  ds = ds.shuffle(16 * batch_size, seed=0)

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds

