from __future__ import absolute_import, division, print_function, unicode_literals
import ipdb

import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

import io_manager
from MyModel import MyModel

# PARAMETERS
dataset_dir = 'data'
output_dir = 'out'
models_dir = 'models'
BATCH_SIZE=32
EPOCHS = 10
print("[Info] Tensorflow is using GPU: {}".format(tf.test.is_gpu_available()))

# LOADING DATASET
print("[Info] Loading Dataset...")
# IF FIRST TIME, RUN THESE LINES:
# x_train, x_val, x_test, y_train, y_val = io_manager.load_dataset_from_images(dataset_dir)
# io_manager.save_dataset_npz(x_train, x_val, x_test, y_train, y_val, 'preprocessed')

# IF ALREADY SAVED NPZ, RUN THIS LINE
x_train, x_val, x_test, y_train, y_val = io_manager.load_dataset_from_npz(dataset_dir)
print("[Info] Dataset loaded.")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

# BUILDING MODEL
print("[Info] Creating the model")
#model = VGG16()
#print(model.summary())
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

# BEGIN LEARNING
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
    save_predictions_csv(predictions, dataset_dir, output_dir)

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
