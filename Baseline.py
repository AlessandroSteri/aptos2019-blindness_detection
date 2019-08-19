import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16

class MultiLayerPerceptron(Model):
  def __init__(self):
    super(MultiLayerPerceptron, self).__init__()
    self.flatten = Flatten()
    self.d1 = Dense(256, activation='relu')
    self.d2 = Dense(128, activation='sigmoid')
    self.d3 = Dense(32, activation='tanh')
    self.d4 = Dense(6, activation='softmax')

  def call(self, x, is_training=True):
    x = tf.cast(x, 'float32')
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    return x

class Baseline(Model):
  def __init__(self):
    super(Baseline, self).__init__()
    self.conv = Conv2D(64, 3, activation='relu')
    self.pool = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.flatten = Flatten()
    self.d = Dense(6, activation='softmax')

  def call(self, x, is_training=True):
    x = tf.cast(x, 'float32')
    x = self.conv(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.d(x)
    return x
