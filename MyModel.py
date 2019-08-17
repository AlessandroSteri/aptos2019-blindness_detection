import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.vgg = VGG16()
    self.conv1 = Conv2D(64, 3, activation='relu')
    self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv2 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d1_2 = Dense(56, activation='sigmoid')
    # self.d1_4 = Dense(28, activation='relu')
    # self.d1_8 = Dense(14, activation='relu')
    self.d2 = Dense(6, activation='softmax')
    self.dropout1 = Dropout(0.5)
    self.dropout2 = Dropout(0.5)
    # self.dropout3 = Dropout(0.4)
    # self.dropout4 = Dropout(0.3)

  def call(self, x, is_training=True):
    x = tf.cast(x, 'float32')
    for layer in self.vgg.layers[:-5]:
        layer.trainable = False
        x = layer(x)
    x = self.conv1(x)
    x = self.dropout1(x, training=is_training)
    x = self.pool1(x)
    x = self.dropout2(x, training=is_training)
    x = self.conv2(x)
    # x = self.dropout3(x, training=is_training)
    x = self.flatten(x)
    # x = self.dropout4(x, training=is_training)
    x = tf.concat([self.d1(x), x], -1)
    x = self.d1_2(x)
    # x = self.d1_4(x)
    return self.d2(x)
