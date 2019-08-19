import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import preprocessing as pre
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
