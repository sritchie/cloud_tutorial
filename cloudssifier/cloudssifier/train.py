"""Train a classifier on MNIST"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib

FLAGS = flags.FLAGS

default_data_path = '../../data/mnist.npz'

flags.DEFINE_integer('epochs', 10, 'number of training epochs.')
flags.DEFINE_integer('batch_size', 64, 'training batch size.')
flags.DEFINE_string('data_path', default_data_path, 'path to mnist npz file.')
flags.DEFINE_string('job-dir',  '',  'Catch job directory passed by cloud.')

def get_mnist(data_path):
  """
  Download if necesairy, process and return MNIST data.
  Inputs:
    data_dir - Path to data, or where data should be placed.

  Outputs:
    data - Tuple of tuples, consisting of formatted train and test data.
  """
  logging.info('Attempting to get data from %s' % data_path)
  with tf.gfile.GFile(data_path, mode='rb') as f:
    data = np.load(f)

  x_train, y_train_cold = data['x_train'], data['y_train']
  x_test, y_test_cold = data['x_test'], data['y_test']

  x_train = x_train.reshape(-1, 28*28)/255.0
  x_train = x_train.astype(np.float32)
  x_test = x_test.reshape(-1, 28*28)/255.0
  x_test = x_test.astype(np.float32)

  l_train = len(y_train_cold)
  l_test = len(y_test_cold)

  y_train = np.zeros((l_train, 10), dtype=np.float32)
  y_test = np.zeros((l_test, 10), dtype=np.float32)
  y_train[np.arange(l_train),y_train_cold] = 1
  y_test[np.arange(l_test),y_test_cold] = 1

  data = ((x_train, y_train), (x_test, y_test))

  return data

def get_model():
  """
  Build and compile a vanilla fully connected net with SGD optimizer.
  Outputs:
    model - Compiled model
  """
  logging.info('Building model')

  # Model
  model = Sequential()
  model.add(Dense(256, input_shape=(28*28,)))
  model.add(Activation('relu'))
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dense(10))
  model.add(Activation('softmax'))

  # Optimizer
  optimizer = SGD(lr=0.1)

  # Compile
  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

  return model


def train_and_validate():
  """Run main training loop and report final accuracy"""

  logging.info('GPU INFO:')
  logging.info(device_lib.list_local_devices())
  logging.info('Starting model training')

  EPOCHS = FLAGS.epochs
  BATCH_SIZE = FLAGS.batch_size
  DATA_PATH = FLAGS.data_path

  (x_train, y_train), (x_test, y_test) = get_mnist(DATA_PATH)

  model = get_model()

  model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

  logging.info('Validating model.')

  sess = K.get_session()

  train_preds = np.argmax(model(x_train).eval(session=sess), axis=1)
  train_labels = np.argmax(y_train, axis=1)
  train_acc = np.mean(train_preds == train_labels)

  test_preds = np.argmax(model(x_test).eval(session=sess), axis=1)
  test_labels = np.argmax(y_test, axis=1)
  test_acc = np.mean(test_preds == test_labels)

  print('Train acc: %f, Test acc: %f' % (train_acc, test_acc))

def main(argv):
  logging.info('Running main.')
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  else:
    train_and_validate()

if __name__ == '__main__':
  app.run(main)
