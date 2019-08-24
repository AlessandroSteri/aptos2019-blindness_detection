from __future__ import absolute_import, division, print_function, unicode_literals
import ipdb
import argparse
import io
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, batch_normalization
# from tensorflow.keras import Model
# from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
import io_manager
from MyModel import MyModel, MyModelMultihead, MultiheadAttentive, MultiheadAttentiveBiLSTM, MultiheadResNet, MultiheadAttentiveNoVGG, MultiheadAttentiveBiLSTMNoVGG
from Baseline import Baseline, MultiLayerPerceptron

from utils import id_gen, mkdir

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument("log_info", type=str, help="Info to log.")
cmdLineParser.add_argument("--preproc", dest='preproc', type=int, choices=[1, 2], help="Preprocessing 1 or 2.", default=2)
cmdLineParser.add_argument('--traintest', dest='traintest', type=float, help="Split in 1 digit float.", default=0.30)
cmdLineParser.add_argument('--epochs', dest='epochs', type=int, help="Number of epochs.", default=30)
cmdLineParser.add_argument('--ksplits', dest='ksplits', type=int, help="Number of splits for k fold val.", default=10)
cmdLineParser.add_argument('--dataaug', dest='dataaug', type=int, help="Factor of data augmentation.", default = 1)
cmdLineArgs = cmdLineParser.parse_args()
# PARAMETERS
log_info = cmdLineArgs.log_info
PREPROCESSING = cmdLineArgs.preproc
TRAIN_TEST_SPLIT = cmdLineArgs.traintest
EPOCHS = cmdLineArgs.epochs
NSPLITS = cmdLineArgs.ksplits
DATAAUG_FACTOR = cmdLineArgs.dataaug
dataset_dir = 'data'
output_dir = 'out'
models_dir = 'models'
tensorboard_dir = 'tensorboard'
mkdir(tensorboard_dir)
BATCH_SIZE = 32
THRESHOLD = 75.0    # Save only models with accuracy above the threshold

# Uncomment to force CPU run
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# DEVELOPMENT ENV: set seed to have reproducible result
from tensorflow.random import set_seed
set_seed(3)

ID = id_gen() #Why not datetime?
tf_log_name = "{}_{}".format(ID, log_info)
tb_base_dir = os.path.join(tensorboard_dir, tf_log_name)
mkdir(tb_base_dir)

print("[Info] Tensorflow is using GPU: {}".format(tf.test.is_gpu_available()))

# LOADING DATASET
print("[Info] Loading Dataset...")
# IF FIRST TIME, RUN THESE LINES:
# x_train, y_train = io_manager.load_dataset_from_images(dataset_dir)
# io_manager.save_dataset_npz(x_train, y_train, 'preprocessed_1')

# IF ALREADY SAVED NPZ, RUN THIS LINE
if PREPROCESSING==1:
    x_train, y_train = io_manager.load_dataset_from_npz('preprocessed_1')
else:
    x_train, y_train = io_manager.load_dataset_from_npz('preprocessed_2')

print("[Info] Dataset loaded.")

# SPLIT TRAIN/TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=TRAIN_TEST_SPLIT)
print("[Info] Data splitted in train ({}) and test ({}).".format(len(y_train), len(y_test)))

#DATA AUGMENTATION
classes = [0,1,2,3,4]
print("[Info] Training set data augmentation. Tot: {} images".format(len(y_train)))
for c in classes:
    print("    Class {}: {}".format(c, len(y_train[y_train==c])))
num_instances = [DATAAUG_FACTOR*i for i in [149, 146, 139, 159, 94]]
#in this way tot is [1400, 400, 850, 300, 300]
augX, augY = io_manager.data_augmentation(  x_train, y_train,
                                            labels=classes,
                                            instances=num_instances)
x_train = np.concatenate((x_train, augX))
y_train = np.concatenate((y_train, augY))
print("[Info] Training set data augmentation. Tot: {} images".format(len(y_train)))
for c in classes:
    print("    Class {}: {}".format(c, len(y_train[y_train==c])))

# BUILDING MODEL
print("[Info] Creating the model")
# model = Baseline()
# model = MultiLayerPerceptron()
# model = MyModel()
# model = MyModelMultihead()
# model = MultiheadAttentive()
model = MultiheadAttentiveBiLSTM()
# broken
# model = MultiheadResNet()
# model = MultiheadAttentiveNoVGG()
# model = MultiheadAttentiveBiLSTMNoVGG()



plot_model_file = os.path.join(tb_base_dir, 'model.png')
plot_model(model, to_file=plot_model_file, show_shapes=True, show_layer_names=True)

# TensorBoard summary
summary_writer = tf.summary.create_file_writer(tb_base_dir)
cm_dir = os.path.join(tb_base_dir, 'cm')
file_writer = tf.summary.create_file_writer(cm_dir)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy() #model_cohen_kappa
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

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
    p = tf.math.argmax(predictions, -1)
    return p

@tf.function
def val_step(images, labels):
    predictions = model(images, is_training=False)
    t_loss = loss_object(labels, predictions)
    val_loss(t_loss)
    val_accuracy(labels, predictions)

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

# def log_confusion_matrix(labels, predictions, epoch):
    # pass


# BEGIN LEARNING
best_accuracy = 0.0

# TO DO: USE THIS VAL_DS AS TEST SET
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(BATCH_SIZE)
# val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
k_fold = KFold(n_splits=NSPLITS, shuffle=True)

log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

epoch = 0
while(epoch<EPOCHS):
    for i_fold, (train_indices, val_indices) in enumerate(k_fold.split(x_train)):
        # Create fold split in train e validation sets
        if(epoch>=EPOCHS):
            break
        k_fold_x_train = x_train[train_indices]
        k_fold_y_train = y_train[train_indices]
        k_fold_x_val = x_train[val_indices]
        k_fold_y_val = y_train[val_indices]
        train_ds = tf.data.Dataset.from_tensor_slices((k_fold_x_train, k_fold_y_train)).shuffle(100000).batch(BATCH_SIZE)
        val_ds = tf.data.Dataset.from_tensor_slices((k_fold_x_val, k_fold_y_val)).batch(BATCH_SIZE)

        # Train, Validation and Test steps
        print("[Info] Starting Epoch {}/{} --> Fold {}/{}, ID: {}".format(epoch + 1, EPOCHS, i_fold+1, NSPLITS, ID))
        for images, labels in train_ds:
            train_step(images, labels)
        for val_images, val_labels in val_ds:
            val_step(val_images, val_labels)
        # model.summary()

        fold_predictions = []
        fold_labels = []
        for test_images, test_labels in test_ds:
            # ipdb.set_trace()
            # p =
            p = test_step(test_images, test_labels)
            fold_predictions = tf.concat([fold_predictions, p], -1)
            fold_labels = tf.concat([fold_labels, test_labels], -1)

        # confusion_matrix_summary(test_labels, p, [0,1,2,3,4], optimizer.iterations)
        con_mat = tf.math.confusion_matrix(labels=fold_labels, predictions=fold_predictions).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        classes = [0,1,2,3,4]
        con_mat_df = pd.DataFrame(con_mat_norm,
                        index = classes,
                        columns = classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        image = tf.expand_dims(image, 0)

        # Log the confusion matrix as an image summary.
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch) #(epoch*NSPLITS)+ i_fold)

        # Always update best accuracy
        if test_accuracy.result() > best_accuracy:
            best_accuracy = test_accuracy.result()

        # Save only models sufficiently good
        if best_accuracy > THRESHOLD:
            predictions = predict()
            io_manager.save_predictions_csv(predictions, dataset_dir, output_dir)

            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S")
            model_name = '{}_model_{}_{}'.format(timestampStr, int(best_accuracy*10000), epoch)
            model.save_weights(os.path.join(models_dir, model_name, 'model_weights'), save_format='tf')
            print('[Info] Saved model.')

        template = '\tLoss: {}, Accuracy: {}\n\tVal Loss: {}, Val Accuracy: {}\n\tTest Loss: {}, Test Accuracy: {},\n\tBest Accuracy: {}'
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=optimizer.iterations)
            tf.summary.scalar('val_loss', val_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('val_accuracy', val_accuracy.result(), step=optimizer.iterations)
            tf.summary.scalar('test_loss', test_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('test_accuracy', test_accuracy.result(), step=optimizer.iterations)
            tf.summary.scalar('best_accuracy', best_accuracy, step=optimizer.iterations)
        print(template.format(train_loss.result(),
                            train_accuracy.result()*100,
                            val_loss.result(),
                            val_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100,
                            best_accuracy*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        epoch = epoch + 1
