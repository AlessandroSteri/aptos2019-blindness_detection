from __future__ import absolute_import, division, print_function, unicode_literals


# uncomment to force cpu exeution
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import ipdb

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

#########################
# APTOS Dataset - BEGIN #
#########################
import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())
# input("Press any button to continue...")
dataset_dir = 'data'
output_dir = 'out'
models_dir = 'models'

def load_dataset_from_images():
    # Path variables
    train_dir   = os.path.join(dataset_dir, 'train_images')
    test_dir    = os.path.join(dataset_dir, 'test_images')
    IMG_SIZE = 224      #224x244 is the size of ImageNet

    # Function to open image and resize it
    def load_image_resized(image_path, desired_size=IMG_SIZE):
        im = Image.open(image_path)
        return im.resize((desired_size, )*2, resample=Image.LANCZOS)

    # Load csv file sa pandas dataframe
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    x_train = np.empty((train_df.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    x_test = np.empty((test_df.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Debug
    print("Training set Shape " + str(train_df.shape))
    print("Test set shape " + str(test_df.shape))
    # Training set label distribution
    plt.show(train_df['diagnosis'].hist())

    # Load training data
    for i, image_id in enumerate(tqdm(train_df['id_code'])):
        x_train[i, :, :, :] = load_image_resized(
            os.path.join(train_dir, "{}.png".format(image_id))
        )
    y_train = train_df['diagnosis'].values  #Labels

    # Load test data
    for i, image_id in enumerate(tqdm(test_df['id_code'])):
        x_test[i, :, :, :] = load_image_resized(
            os.path.join(test_dir, "{}.png".format(image_id))
        )
    # THERE ARE NO LABELS FOR TEST DATA

    #Debug
    print("Size Train Set: " + str(x_train.shape))
    print("Size Test Set: " + str(x_test.shape))

    # Split dataset in Train and Validation set
    x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.15)
    x_train, x_val, x_test = x_train / 255.0, x_val/ 255.0, x_test / 255.0
    return x_train, x_val, x_test, y_train, y_val

def load_dataset_from_npz():
    x_train_filepath = os.path.join(dataset_dir, 'x_train.npz')
    x_test_filepath  = os.path.join(dataset_dir, 'x_test.npz')
    x_val_filepath   = os.path.join(dataset_dir, 'x_val.npz')
    y_train_filepath = os.path.join(dataset_dir, 'y_train.npz')
    y_val_filepath   = os.path.join(dataset_dir, 'y_val.npz')
    x_train = np.load(x_train_filepath)['arr_0']
    x_test = np.load(x_test_filepath)['arr_0']
    x_val = np.load(x_val_filepath)['arr_0']
    y_train = np.load(y_train_filepath)['arr_0']
    y_val = np.load(y_val_filepath)['arr_0']
    return x_train, x_val, x_test, y_train, y_val

def save_dataset_npz(x_train, x_val, x_test, y_train, y_val):
    np.savez_compressed(os.path.join(dataset_dir, 'x_train.npz'), x_train)
    np.savez_compressed(os.path.join(dataset_dir, 'x_val.npz'), x_val)
    np.savez_compressed(os.path.join(dataset_dir, 'x_test.npz'), x_test)
    np.savez_compressed(os.path.join(dataset_dir, 'y_train.npz'), y_train)
    np.savez_compressed(os.path.join(dataset_dir, 'y_val.npz'), y_val)

print("[Info] Loading Dataset...")
# IF FIRST TIME, RUN THESE LINES:
# x_train, x_val, x_test, y_train, y_val = load_dataset_from_images()
# save_dataset_npz(x_train, x_val, x_test, y_train, y_val)

# IF ALREADY SAVED NPZ, RUN THIS LINE
x_train, x_val, x_test, y_train, y_val = load_dataset_from_npz()
print("[Info] Dataset loaded.")
#########################
#  APTOS Dataset - END  #
#########################

# Add a channels dimension  - NOT NEEDED BECAUSE WE ALREADY HAVE THE CHANNEL AS THIRD COMPONENT
# x_train = x_train[..., tf.newaxis]
# x_val= x_val[..., tf.newaxis]

# Model Parameters
BATCH_SIZE=32
EPOCHS = 10000
print("[Info] Tensorflow is using GPU: {}".format(tf.test.is_gpu_available()))

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

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

# Create an instance of the model
# ipdb.set_trace()
print("[Info] Creating the model")
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, is_training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

@tf.function
def pred_step(images):
    return model(images, is_training=False)

def predict():
    # Make predictions
    print("[Info] Making predictions...")
    # print("Test set shape " + str(x_test.shape))
    # ipdb.set_trace()
    test_dir    = os.path.join(dataset_dir, 'test_images')
    test_df  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

    predictions = []
    for images in test_ds:
        predictions = tf.concat([predictions, tf.math.argmax(pred_step(images), -1)], -1)
    return predictions

def save_predictions_csv(predictions):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S")
    out_file = "{}_predictions.csv".format(timestampStr)
    test_dir = os.path.join(dataset_dir, 'test_images')
    test_df  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    test_df['diagnosis']=predictions
    test_df.to_csv(os.path.join(output_dir, "prediction.csv"), index=False)

best_accuracy = 0.7945

for epoch in range(EPOCHS):
  print("[Info] Starting Epoch --> {}".format(epoch + 1))
  for images, labels in train_ds:
    train_step(images, labels)
  for test_images, test_labels in val_ds:
    test_step(test_images, test_labels)

  template = '\tLoss: {}, Accuracy: {}\n\tTest Loss: {}, Test Accuracy: {}\nBest Accuracy: {}'
  print(template.format(train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        best_accuracy*100,
                        test_accuracy.result()*100))

  if test_accuracy.result() > best_accuracy:
    best_accuracy = test_accuracy.result()
    predictions = predict()
    save_predictions_csv(predictions)

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S")
    model_name = '{}_model_{}_{}'.format(timestampStr, int(best_accuracy*10000), epoch)
    model.save_weights(os.path.join(models_dir, model_name, 'model_weights'), save_format='tf')
    print('Predicted with accuracy {} on Validation Set'.format(best_accuracy))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()


