import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocessing as pre

def load_dataset_from_images(dataset_dir):
    # Path variables
    train_dir   = os.path.join(dataset_dir, 'train_images')
    test_dir    = os.path.join(dataset_dir, 'test_images')
    IMG_SIZE = 224      #224x244 is the size of ImageNet
    TRAIN_VAL_SPLIT = 0.30

    # Function to open image and resize it
    def load_image_resized(image_path, desired_size=IMG_SIZE):
        return pre.preprocess(image_path)
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
        img = x_train[i]
        plt.imshow(img)
        plt.show()
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
    x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=TRAIN_VAL_SPLIT)
    x_train, x_val, x_test = x_train / 255.0, x_val/ 255.0, x_test / 255.0
    return x_train, x_val, x_test, y_train, y_val

def get_ImageGenerator():
    gen = ImageDataGenerator(width_shift_range=0.05,    #percentage [-0.2,+0.2] of image size
                             height_shift_range=0.05,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='constant', cval=0,
                             rotation_range=90,
                             brightness_range=[-0.20, +0.20])
    return gen

def data_augmentation(X, y=None, labels=[], instances=[100]):
    # 2 mode: no label -> labels=[] and instances=[N], return sample N instances from X
    #         by label -> labels=[a,b,c] and instances=[Na,Nb,Nc], return sample of each label
    num_labels = len(labels)
    num_instances = len(instances)
    if num_labels==0 and num_instances!=1:
        raise Exception("[Error] Data Augmentation - Invalid number labels {}, number instances {}".format(num_labels, num_instances))
    if num_labels>0 and (num_labels!=num_instances or y is None):
        raise Exception("[Error] Data Augmentation - Invalid number labels {} != number instances {}".format(num_labels, num_instances))
    if num_labels==0:
        return data_augmentation_all(X, y, instances[0])
    else:
        return data_augmentation_by_label(X, y, labels, instances)

def data_augmentation_all(X, y=None, instances=100):
    # mode 1: sample from all `X` a given number of elements (`instances`)
    # if y==None, return augY=None
    it = get_ImageGenerator().flow(X, y, batch_size=1)
    augX = np.empty((instances, X.shape[1], X.shape[2], X.shape[3]), dtype=np.uint8)
    if y is None:
        augY = None
    else:
        augY = np.empty((instances), dtype=np.uint8)
    for i in range(instances):
        if y is None:
            image = it.next()[0].astype('uint8')
        else:
            image, label = it.next()
            image = image.astype('uint8')
            augY[i] = label
        augX[i, :, :, :] = image
    return augX, augY

def data_augmentation_by_label(X, y, labels=[0], instances=[100]):
    num_labels=len(labels)
    num_instances=len(instances)
    if num_labels>0 and num_labels!=num_instances:
        raise Exception("[Error] Data Augmentation - Invalid number labels {} != number instances {}".format(num_labels, num_instances))
    # Pre allocate structure
    augX = np.empty((sum(instances), X.shape[1], X.shape[2], X.shape[3]), dtype=np.uint8)
    augY = np.empty((sum(instances)), dtype=np.uint8)
    begin = 0
    for label, num in zip(labels, instances):
        mask = y==label
        augX[begin:begin+num, :, :, :], augY[begin:begin+num] = data_augmentation_all(X[mask], y[mask], num)
        begin = begin + num
    return augX, augY

def load_dataset_from_npz(dataset_dir):
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

def save_dataset_npz(x_train, x_val, x_test, y_train, y_val, dataset_dir):
    np.savez_compressed(os.path.join(dataset_dir, 'x_train.npz'), x_train)
    np.savez_compressed(os.path.join(dataset_dir, 'x_val.npz'), x_val)
    np.savez_compressed(os.path.join(dataset_dir, 'x_test.npz'), x_test)
    np.savez_compressed(os.path.join(dataset_dir, 'y_train.npz'), y_train)
    np.savez_compressed(os.path.join(dataset_dir, 'y_val.npz'), y_val)

def save_predictions_csv(predictions, dataset_dir, output_dir):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S")
    out_file = "{}_predictions.csv".format(timestampStr)
    test_dir = os.path.join(dataset_dir, 'test_images')
    test_df  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    test_df['diagnosis']=predictions
    test_df.to_csv(os.path.join(output_dir, "prediction.csv"), index=False)
