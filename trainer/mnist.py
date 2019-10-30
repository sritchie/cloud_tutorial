"""
This is from cloud-hello-world.
"""
from __future__ import division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import time

print("TF Version:", tf.__version__)

#tf.enable_eager_execution()

batch_size = 32
width = 100
lr = 0.1
steps = 10000
measure_every = 500


def whiten(batch):
  batch['image'] = tf.cast(batch['image'], tf.float32) / 255.
  return batch


dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
dataset = dataset.map(whiten)

measure_dataset = dataset.batch(256)
dataset = dataset.shuffle(1000).repeat().batch(batch_size)


def get_keras_model(activation):
  return tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(width, activation=activation),
      tf.keras.layers.Dense(width, activation=activation),
      tf.keras.layers.Dense(10, activation=None),
  ])


model = get_keras_model('relu')


#@tf.function
def compute_loss(labels, logits):
  #cur_batch_size = tf.cast(labels.shape[0], tf.float32)
  #return tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.SUM) / cur_batch_size
  return tf.reduce_mean(
      tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                      logits,
                                                      from_logits=True))


#@tf.function
def compute_acc(labels, logits):
  cur_batch_size = tf.cast(labels.shape[0], tf.float32)
  predictions = tf.argmax(logits, axis=1)
  return tf.reduce_sum(tf.cast(tf.equal(labels, predictions),
                               tf.float32)) / cur_batch_size


#@tf.function
def compute_batch_loss_acc(images, labels, model, training):
  logits = model(images, training=training)
  loss = compute_loss(labels, logits)
  acc = compute_acc(labels, logits)
  return loss, acc


def compute_loss_acc(dataset, model):
  total_loss = 0
  total_acc = 0.
  total_samples = 0

  for batch in dataset:
    images = batch['image']
    labels = batch['label']
    cur_batch_size = len(labels)
    #logits = model(images, training=True)
    batch_loss, batch_acc = compute_batch_loss_acc(images,
                                                   labels,
                                                   model,
                                                   training=True)
    total_loss += batch_loss * cur_batch_size
    total_acc += batch_acc * cur_batch_size
    total_samples += cur_batch_size

  total_loss /= total_samples
  total_acc /= total_samples
  return total_loss, total_acc


print('width={} lr={}'.format(width, lr))

batch_loss_history = []
batch_acc_history = []
optimizer = tf.keras.optimizers.SGD(lr)
#optimizer = tf.train.GradientDescentOptimizer(lr)


@tf.function  # speeds things up by 1.4x
def train_step(images, labels):
  with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss = compute_loss(labels, logits)

  acc = compute_acc(labels, logits)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss, acc


start = time.time()

for (step, batch) in enumerate(dataset.take(steps)):
  images = batch['image']
  labels = batch['label']

  if step % measure_every == 0:
    train_loss, train_acc = compute_loss_acc(measure_dataset, model)
    print('{}: loss={} acc={}'.format(step, train_loss.numpy(),
                                      train_acc.numpy()))

  loss, acc = train_step(images, labels)
  batch_loss_history.append(loss.numpy())
  batch_acc_history.append(acc.numpy())

elapsed = time.time() - start
print("Took {} seconds".format(elapsed))
