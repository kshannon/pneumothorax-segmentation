import tensorflow as tf  # conda install -c anaconda tensorflow

import keras as K

import numpy as np

from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss

from dataloader import DataGenerator

import matplotlib.pyplot as plt

height = 256
width = 256

validation_data = DataGenerator(img_path="../data/train/*.dcm",
                              rle_csv="../data/train-rle.csv", validation=True,
                              batch_size=100,
                              height=height,
                              width=width,
                              shuffle=True,
                              train_class_one=True)

model = K.models.load_model("pneumothorax.hdf5", custom_objects={"dice_coef": dice_coef, "dice_coef_loss":dice_coef_loss, "soft_dice_coef":soft_dice_coef})

idx = np.random.randint(0, validation_data.__len__())
X, y = validation_data.__getitem__(idx)

pred = model.predict(X)

rows = 4
for ii in range(rows):

    plt.subplot(rows,3,1+ii*3)
    plt.imshow(X[ii,:,:,0])
    if ii==0:
        plt.title("Original image")
    plt.subplot(rows,3,2+ii*3)
    plt.imshow(y[ii,:,:,0])
    if ii==0:
        plt.title("Ground truth")
    plt.subplot(rows,3,3+ii*3)
    plt.imshow(pred[ii,:,:,0])
    if ii==0:
        plt.title("Prediction")

plt.show()
