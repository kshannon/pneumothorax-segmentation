import numpy as np
from dataloader import DataGenerator

import matplotlib.pyplot as plt

height = 1024
width = 1024
rows = 4

validation_data = DataGenerator(img_path="../data/train/*.dcm",
                              rle_csv="../data/train-rle.csv", validation=True,
                              batch_size=rows,
                              height=height,
                              width=width,
                              shuffle=True,
                              train_class_one=True)

idx = np.random.randint(0, validation_data.__len__())
X, y = validation_data.__getitem__(idx)

for ii in range(rows):

    plt.subplot(rows,2,1+ii*2)
    plt.imshow(X[ii,:,:,0])
    if ii==0:
        plt.title("Original image")
    plt.subplot(rows,2,2+ii*2)
    plt.imshow(y[ii,:,:,0])
    if ii==0:
        plt.title("Ground truth")


plt.show()
