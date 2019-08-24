import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LSTM, Bidirectional, Attention

class ResnetBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetBlock, self).__init__(name='')
    f, ff, fff = filters

    self.conv2a = Conv2D(f, (1, 1))
    self.bn2a = BatchNormalization()

    self.conv2b = Conv2D(ff, kernel_size, padding='same')
    self.bn2b = BatchNormalization()

    self.conv2c = Conv2D(fff, (1, 1))
    self.bn2c = BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x = tf.concat([x, input_tensor],-1)
    return tf.nn.relu(x)

