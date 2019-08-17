import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ipdb

def crop_image_from_gray(image,tol=7):
    if image.ndim ==2:
        mask = image>tol
        return image[np.ix_(mask.any(1),mask.any(0))]
    elif image.ndim==3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray_image>tol
        check_shape = image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return image
        else:
            image1=image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            image2=image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            image3=image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            image = np.stack([image1,image2,image3],axis=-1)
        return image

def circle_crop_v2(image):
    """
    Create circular crop around image centre
    """
    # image = cv2.imread(imagepath)
    # image = crop_image_from_gray(image)

    height, width, depth = image.shape
    largest_side = np.max((height, width))
    # image = cv2.resize(image, (largest_side, largest_side))
    # height, width, depth = image.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_image = np.zeros((height, width), np.uint8)
    cv2.circle(circle_image, (x, y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_image)
    # image = crop_image_from_gray(image)

    return image

def load_ben_color(path, sigmaX=10):
    IMG_SIZE = 224
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

def preprocess_test():
    dataset_dir = os.path.join('data', 'train_images')
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))
    for i, filename in enumerate(os.listdir(dataset_dir)):
        filepath = os.path.join(dataset_dir, filename)
        img = load_ben_color(filepath)
        img = circle_crop_v2(img)
        # ax[i].imshow(img)
        # ax[i].axis('off')
        print(filepath)
        print(img.shape)
    # plt.show()

def preprocess(image_path):
    img = load_ben_color(image_path)
    img = circle_crop_v2(img)
    return img
