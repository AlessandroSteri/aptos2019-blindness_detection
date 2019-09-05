import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LSTM, Bidirectional, Attention, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50

from layers import ResnetBlock

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.vgg = VGG16()
    self.conv1 = Conv2D(64, 3, activation='relu')
    self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv2 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()

    self.d1 = Dense(128, activation='relu', use_bias=True)
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid', use_bias=True)
    # self.d1_4 = Dense(28, activation='relu')
    # self.d1_8 = Dense(14, activation='relu')
    self.d2 = Dense(6, activation='softmax', use_bias=True)
    self.dropout1 = Dropout(0.5)
    self.dropout2 = Dropout(0.5)
    # self.dropout3 = Dropout(0.4)
    # self.dropout4 = Dropout(0.3)

  def call(self, x, is_training=True):
    x = tf.cast(x, 'float32')
    x = self.bn(x)
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


class MyModelMultihead(Model):
  def __init__(self):
    super(MyModelMultihead, self).__init__()
    self.vgg = VGG16()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    # self.d1 = Dense(128, activation='relu')
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    # self.d1_4 = Dense(28, activation='relu')
    # self.d1_8 = Dense(14, activation='relu')
    self.d2 = Dense(5, activation='softmax')
    self.dropout_vgg = Dropout(0.5)
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)
    # self.lstm_max_fw = LSTM(128, return_sequences=True)
    # self.lstm_max_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    # self.bilstm_max = Bidirectional(self.lstm_max_fw, backward_layer=self.lstm_max_bw)
    # self.lstm_avg_fw = LSTM(128, return_sequences=True)
    # self.lstm_avg_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    # self.bilstm_avg = Bidirectional(self.lstm_avg_fw, backward_layer=self.lstm_avg_bw)
    # self.dropout4 = Dropout(0.3)
    self.res_block = ResnetBlock(1, [1, 2, 3])
    self.res_block2 = ResnetBlock(1, [1, 2, 3])
    self.res_block3 = ResnetBlock(1, [1, 2, 3])
    self.res_block4 = ResnetBlock(1, [1, 2, 3])

  def call(self, x, is_training=True):
    # x_original = x
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    for layer in self.vgg.layers[:-5]:
        layer.trainable = False
        x = layer(x)
    x = self.dropout_vgg(x, training=is_training)
    x = self.res_block(x)
    x = self.res_block2(x)
    x = self.res_block3(x)
    x = self.res_block4(x)

    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])
    # x_max = self.bilstm_max(x_max)

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])
    # x_avg = self.bilstm_avg(x_avg)

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    # x = self.d1_4(x)
    return self.d2(x)


class MultiheadAttentive(Model):
  def __init__(self):
    super(MultiheadAttentive, self).__init__()
    self.vgg = VGG16()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    # self.d1 = Dense(128, activation='relu')
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    # self.d1_4 = Dense(28, activation='relu')
    # self.d1_8 = Dense(14, activation='relu')
    self.d2 = Dense(6, activation='softmax')
    self.dropout_vgg = Dropout(0.5)
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)

  def call(self, x, is_training=True):
    # x_original = x
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    for layer in self.vgg.layers[:-5]:
        layer.trainable = False
        x = layer(x)
    x = self.dropout_vgg(x, training=is_training)

    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    return self.d2(x)

class MultiheadAttentiveBiLSTM(Model):
  def __init__(self):
    super(MultiheadAttentiveBiLSTM, self).__init__()
    self.vgg = VGG16()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    self.d2 = Dense(6, activation='softmax')
    self.dropout_vgg = Dropout(0.5)
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)
    self.lstm_max_fw = LSTM(128, return_sequences=True)
    self.lstm_max_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    self.bilstm_max = Bidirectional(self.lstm_max_fw, backward_layer=self.lstm_max_bw)
    self.lstm_avg_fw = LSTM(128, return_sequences=True)
    self.lstm_avg_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    self.bilstm_avg = Bidirectional(self.lstm_avg_fw, backward_layer=self.lstm_avg_bw)
    self.dropout4 = Dropout(0.3)

  def call(self, x, is_training=True):
    # x_original = x
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    for layer in self.vgg.layers[:-5]:
        layer.trainable = False
        x = layer(x)
    x = self.dropout_vgg(x, training=is_training)

    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])
    x_max = self.bilstm_max(x_max)

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])
    x_avg = self.bilstm_avg(x_avg)

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    # x = self.d1_4(x)
    return self.d2(x)

class SimpleResNet(Model):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.resnet.trainable=False
        self.drop03 = Dropout(0.3)
        self.dense1024 = Dense(1024, activation='relu')
        self.dense512 = Dense(512, activation='relu')
        self.dense256 = Dense(256, activation='relu')
        self.glob_avg = GlobalAveragePooling2D(data_format='channels_last')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.dense_last = Dense(5, activation='softmax')
        # self.flatten = Flatten()

    def call(self, x, is_training=True):
        x = tf.cast(x, 'float32')
        x = self.bn1(x)
        x = self.resnet(x)
        x = self.glob_avg(x)
        x = self.drop03(x)
        x = self.dense1024(x)

        x = self.bn2(x)
        x = self.drop03(x)
        x = self.dense512(x)

        x = self.bn3(x)
        x = self.drop03(x)
        x = self.dense256(x)

        x = self.bn4(x)
        x = self.drop03(x)
        # x = self.flatten(x)
        x = self.dense_last(x)
        return x

class MultiheadResNet(Model):
  def __init__(self):
    super(MultiheadResNet, self).__init__()
    self.vgg = VGG16()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    self.d2 = Dense(6, activation='softmax')
    self.dropout_vgg = Dropout(0.5)
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)
    self.res_block = ResnetBlock(1, [1, 2, 3])
    self.res_block2 = ResnetBlock(1, [1, 2, 3])
    self.res_block3 = ResnetBlock(1, [1, 2, 3])
    self.res_block4 = ResnetBlock(1, [1, 2, 3])

  def call(self, x, is_training=True):
    # x_original = x
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    for layer in self.vgg.layers[:-5]:
        layer.trainable = False
        x = layer(x)
    x = self.dropout_vgg(x, training=is_training)
    x = self.res_block(x)
    x = self.res_block2(x)
    x = self.res_block3(x)
    x = self.res_block4(x)

    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    return self.d2(x)



class MultiheadAttentiveBiLSTMNoVGG(Model):
  def __init__(self):
    super(MultiheadAttentiveBiLSTMNoVGG, self).__init__()
    # self.vgg = VGG16()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    self.d2 = Dense(6, activation='softmax')
    # self.dropout_vgg = Dropout(0.5)
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)
    self.lstm_max_fw = LSTM(128, return_sequences=True)
    self.lstm_max_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    self.bilstm_max = Bidirectional(self.lstm_max_fw, backward_layer=self.lstm_max_bw)
    self.lstm_avg_fw = LSTM(128, return_sequences=True)
    self.lstm_avg_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
    self.bilstm_avg = Bidirectional(self.lstm_avg_fw, backward_layer=self.lstm_avg_bw)
    self.dropout4 = Dropout(0.3)

  def call(self, x, is_training=True):
    # x_original = x
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    # for layer in self.vgg.layers[:-5]:
        # layer.trainable = False
        # x = layer(x)
    # x = self.dropout_vgg(x, training=is_training)

    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])
    x_max = self.bilstm_max(x_max)

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])
    x_avg = self.bilstm_avg(x_avg)

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    # x = self.d1_4(x)
    return self.d2(x)

class MultiheadAttentiveNoVGG(Model):
  def __init__(self):
    super(MultiheadAttentiveNoVGG, self).__init__()
    self.attention = Attention()
    self.conv_max = Conv2D(64, 3, activation='relu')
    self.dropout_max = Dropout(0.5)
    self.conv_avg = Conv2D(64, 3, activation='relu')
    self.dropout_avg = Dropout(0.5)
    self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    self.conv_max_2 = Conv2D(32, 3, activation='relu')
    self.conv_avg_2 = Conv2D(32, 3, activation='relu')
    self.d_max = Dense(128, activation='relu')
    self.d_avg = Dense(128, activation='relu')
    self.flatten = Flatten()
    self.bn = BatchNormalization()
    self.d1_2 = Dense(56, activation='sigmoid')
    self.d2 = Dense(6, activation='softmax')
    self.dropout_max = Dropout(0.5)
    self.dropout_avg = Dropout(0.5)
    self.dropout = Dropout(0.5)

  def call(self, x, is_training=True):
    x = tf.cast(x, 'float32')
    x = self.bn(x)
    # MAX
    x_max = self.conv_max(x)
    x_max = self.dropout_max(x_max, training=is_training)
    x_max = self.pool_max(x_max)
    x_max = self.conv_max_2(x_max)
    x_max = self.d_max(x_max)
    x_max = tf.reshape(x_max, [-1, 16, 128])

    # AVG
    x_avg = self.conv_avg(x)
    x_avg = self.dropout_avg(x_avg, training=is_training)
    x_avg = self.pool_avg(x_avg)
    x_avg = self.conv_avg_2(x_avg)
    x_avg = self.d_avg(x_avg)
    x_avg = tf.reshape(x_avg, [-1, 16, 128])

    # COMBINE
    x = self.attention([x_max, x_avg])
    x = tf.concat([x_max, x_avg, x], -1)
    x = self.dropout(x, training=is_training)
    x = self.flatten(x)
    x = self.d1_2(x)
    # x = self.d1_4(x)
    return self.d2(x)


# class MultiheadAttentiveBiLSTMVggResNet(Model):
#   def __init__(self):
#     super(MultiheadAttentiveBiLSTMVggResNet, self).__init__()
#     self.vgg = VGG16()
#     self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#     self.resnet.trainable=False
#     self.attention = Attention()
#     self.conv_max = Conv2D(64, 3, activation='relu')
#     self.dropout_max = Dropout(0.5)
#     self.conv_avg = Conv2D(64, 3, activation='relu')
#     self.dropout_avg = Dropout(0.5)
#     self.pool_max = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
#     self.pool_avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
#     self.conv_max_2 = Conv2D(32, 3, activation='relu')
#     self.conv_avg_2 = Conv2D(32, 3, activation='relu')
#     self.d_max = Dense(128, activation='relu')
#     self.d_avg = Dense(128, activation='relu')
#     self.flatten = Flatten()
#     self.bn = BatchNormalization()
#     self.d1_2 = Dense(56, activation='sigmoid')
#     self.d2 = Dense(5, activation='softmax')
#     self.dropout_vgg = Dropout(0.5)
#     self.dropout_max = Dropout(0.5)
#     self.dropout_avg = Dropout(0.5)
#     self.dropout = Dropout(0.5)
#     self.lstm_max_fw = LSTM(128, return_sequences=True)
#     self.lstm_max_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
#     self.bilstm_max = Bidirectional(self.lstm_max_fw, backward_layer=self.lstm_max_bw)
#     self.lstm_avg_fw = LSTM(128, return_sequences=True)
#     self.lstm_avg_bw = LSTM(128, return_sequences=True, go_backwards=True, activation='relu')
#     self.bilstm_avg = Bidirectional(self.lstm_avg_fw, backward_layer=self.lstm_avg_bw)
#     self.dropout4 = Dropout(0.3)
#
#     self.conv_res = Conv2D(64, 3, activation='relu')
#     self.dropout_res = Dropout(0.5)
#     self.pool_res = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
#
#   def call(self, x, is_training=True):
#     # x_original = x
#     x = tf.cast(x, 'float32')
#     x_vgg = self.bn(x)
#     x_res = self.bn(x)
#     for layer in self.vgg.layers[:-5]:
#         layer.trainable = False
#         x_vgg = layer(x_vgg)
#     x_vgg = self.dropout_vgg(x_vgg, training=is_training)
#     x_res = self.resnet(x_res)
#     x_res = self.dropout_res(x_res)
#     x_res = self.conv_res(x_res)
#     x_res = self.pool_res(x_res)
#
#     # MAX
#     x_max = self.conv_max(x)
#     x_max = self.dropout_max(x_max, training=is_training)
#     x_max = self.pool_max(x_max)
#     x_max = self.conv_max_2(x_max)
#     x_max = self.d_max(x_max)
#     x_max = tf.reshape(x_max, [32, 16, -1])
#     x_max = self.bilstm_max(x_max)
#
#     # AVG
#     x_avg = self.conv_avg(x)
#     x_avg = self.dropout_avg(x_avg, training=is_training)
#     x_avg = self.pool_avg(x_avg)
#     x_avg = self.conv_avg_2(x_avg)
#     x_avg = self.d_avg(x_avg)
#     x_avg = tf.reshape(x_avg, [32, 16, -1])
#     x_avg = self.bilstm_avg(x_avg)
#
#     # COMBINE
#     x = self.attention([x_max, x_avg])
#     x_res = tf.reshape(x_res, [32, 16, -1])
#     x = tf.concat([x_max, x_avg, x, x_res], -1)
#     x = self.dropout(x, training=is_training)
#     x = self.flatten(x)
#     x = self.d1_2(x)
#     # x = self.d1_4(x)
#     return self.d2(x)
#
