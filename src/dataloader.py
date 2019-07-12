#!/usr/bin/env python
# coding: utf-8

import numpy as np
import keras as K
import glob
import os
import pydicom
import pandas as pd

class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, im_path, rle_csv, testing=False, batch_size=32, shuffle=True):
        """
        Initialization
        """
        self.batch_size = batch_size

        all_paths = np.array(sorted(glob.glob(im_path)))

        np.random.seed(816)
        idxList = np.arange(len(all_paths))  # List of file indices
        randomIdx = np.random.random(len(all_paths))  # List of random numbers
        # Random number go from 0 to 1. So anything above
        # self.train_split is in the validation list.
        trainIdx = idxList[randomIdx < 0.85]

        validateIdx = idxList[randomIdx >= 0.85]

        if testing:
            self.im_paths = all_paths[validateIdx]
        else:
            self.im_paths = all_paths[trainIdx]

        self.im_path = im_path
        self.mask_df = pd.read_csv(rle_csv, index_col='ImageId')

        self.height = 1024
        self.width = 1024

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.im_paths) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.im_paths))]

        # Find list of IDs
        list_IDs_im = [self.im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def rle2mask(self, rle):
        """
        Convert run-length encoding string to image mask
        """

        mask= np.zeros(self.width * self.height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]

        return mask.reshape(self.width, self.height).T  # Because mask is rotated

    def data_generation(self, list_IDs_im):
        """
        Generates data containing batch_size samples
        """

        X = np.empty((len(list_IDs_im), self.height, self.width, 1))
        y = np.empty((len(list_IDs_im), self.height, self.width, 1))
        num_masks_array = np.zeros(len(list_IDs_im))

        # Generate data
        for idx, im_path in enumerate(list_IDs_im):

            im = pydicom.dcmread(im_path).pixel_array
            X[idx, :, :, 0] = (im - np.mean(im)) / np.std(im)

            img_name = os.path.splitext(im_path)[0].split("/")[-1]

            num_masks = self.mask_df[self.mask_df .index == img_name].values.shape[0]

            num_masks_array[idx] = num_masks

            if num_masks > 1:

                y[idx, :, :, 0] = np.zeros((self.height, self.width))
                for msk_idx in range(num_masks):
                    rle_string = self.mask_df[self.mask_df .index == img_name].values[msk_idx][0]
                    y[idx, :, :, 0] += self.rle2mask(rle_string)

            else:

                rle_string = self.mask_df[self.mask_df.index == img_name].values[0][0]

                if rle_string == " -1" or rle_string == " -1":
                    y[idx, :, :, 0] = np.zeros((self.height, self.width))

                    num_masks_array[idx] = 0
                else:
                    y[idx, :, :, 0] = self.rle2mask(rle_string)


        y[y>1] = 1  # If mask value > 1, then it is 1.

        return X, y

if __name__ == "__main__":

    training_data = DataGenerator(im_path='dicom-images-train/*/*/*.dcm', rle_csv="train-rle.csv", testing=False, batch_size=64,shuffle=False)
    images, masks = training_data.__getitem__(1)
