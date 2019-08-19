import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import io_manager
import ipdb

dir = 'data'

X, x_test, x_no_label, y, y_test = io_manager.load_dataset_from_npz(dir)
ipdb.set_trace()
# img1 = Image.open('eye1.png')
# img1 = img1.resize((224, 224))
# img2 = Image.open('eye2.png')
# img2 = img2.resize((224, 224))
# data1 = np.asarray(img1)
# data2 = np.asarray(img2)
# X = np.array([data1, data2])
# y = np.array([1, 2])

NUM_INSTANCES=12
X, y = io_manager.data_augmentation(X, y, instances=[NUM_INSTANCES])
# X, _ = io_manager.data_augmentation(X, instances=[NUM_INSTANCES])
ROWS=math.floor(math.sqrt(NUM_INSTANCES))
COLS=math.ceil(math.sqrt(NUM_INSTANCES))
fig, axs = plt.subplots(ROWS,COLS)
for i,(image, label) in enumerate(zip(X, y)):
    axs[i//COLS][i%COLS].imshow(image)
    print("{} -> {} {} -> {}".format(i, i//COLS, i%ROWS, label))
plt.show()
