#!/usr/bin/env python
# coding: utf-8

import numpy as np
import keras as K
import glob
import os
import pydicom
import pandas as pd
import PIL
from PIL import Image
import gdcm
import cv2
import sys

class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, img_path, rle_csv, validation=False, batch_size=32, height=1024, width=1024, shuffle=True, train_class_one=False):
        """
        Initialization
        """

        self.img_path = img_path
        self.rle_df = pd.read_csv(rle_csv, index_col='ImageId')
        self.train_class_one = train_class_one

        all_paths = np.array(sorted(glob.glob(self.img_path)))
        if validation:
            self.img_paths = self.test_train_split(all_paths)
        else:
            self.img_paths = self.test_train_split(all_paths,validation=False)

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.img_paths))]
        # Find list of IDs
        list_IDs_im = [self.img_paths[k] for k in indexes]
        # validate data integrity
        data_dict = self.data_validator(list_IDs_im)
        # Generate data
        X, y = self.data_generation(data_dict) #change to sending a dict (data_dict) vs a list of paths

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def rle2mask(self, rle):
        """
        Convert run-length encoding string to image mask
        """
        mask= np.zeros(1024 * 1024)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0

        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 1 #1 for white pixel, 0 is a black pixel
            current_position += lengths[index]

        img = mask.reshape(1024, 1024).T
        if self.width != 1024 and self.height != 1024:
            resized_img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
            return resized_img
        else:
            return img


    def test_train_split(self,all_paths,validation=True):
        """
        1. If train_class_one is True then only return class
        2. If testing flag is passed split data into test/train splits
        """

        if self.train_class_one:
            # make a copy of the mask_df, find the class 1's and make a list of their dcm names
            class1_df = self.rle_df.copy()
            class1_df = class1_df.reset_index()
            class1_df['ImageId'] = class1_df['ImageId'].apply(lambda x: '../data/train/' + x + '.dcm')
            class1_df = class1_df[class1_df[' EncodedPixels'] != ' -1']
            class1_list = class1_df['ImageId'].tolist()
            assert len(class1_list) == 3286 #num of known class 1

            mask = np.isin(all_paths, class1_list) #create a mask to screen all_paths ndarray for only class 1
            assert len(set(class1_list)) == (mask == 1).sum() #check unique class 1's in DF equal same in train dir should be: 2379
            # total train data count - class 1 duplicates via set(ImageID) should equal the known class 1 count - class_1 path mask
            assert self.rle_df.shape[0] - len(set(self.rle_df.index.tolist())) == 3286 - (mask == 1).sum()

            class1_paths = all_paths[mask]
            assert sorted(class1_paths.tolist()) == sorted(set(class1_list)) #ensure class1_paths masks is correct

            all_paths = class1_paths

        np.random.seed(816)
        idxList = np.arange(len(all_paths))  # List of file indices
        randomIdx = np.random.random(len(all_paths))  # List of random numbers
        # Random number go from 0 to 1. So anything above
        # self.train_split is in the validation list.
        trainIdx = idxList[randomIdx < 0.85]
        validateIdx = idxList[randomIdx >= 0.85]

        if validation == False:
            return all_paths[trainIdx]
        else:
            return all_paths[validateIdx]


    def data_validator(self,dcm_paths):
        """
        1. Ensure all dcm imgs in folder exist in the CSV.
        2. Provide ability to filter DICOM images by class (e.g. RLE of '-1')
        3. Return a dict object with a key[dcm_name] : value[list of string RLEs]
        """

        mask = {}
        for n, _id in enumerate(dcm_paths):
            image_id = (_id.split('/')[-1][:-4])
            if image_id in self.rle_df.index:
                if type(self.rle_df.loc[image_id, ' EncodedPixels'] ) == str:
                    mask[image_id] = [self.rle_df.loc[image_id, ' EncodedPixels']]
                else:
                    mask[image_id]= self.rle_df.loc[image_id, ' EncodedPixels'].values.tolist()
            else:
                continue #missing dcm file!
        # if self.train_class_one == True:
        #     return {k:v for k, v in mask.items() if type(v) != '-1' or type(v) != ' -1'}
        # else:
        return mask


    def data_generation(self, data_dict):
        """
        Generates data containing batch_size samples
        """
        img_count = len(data_dict.values())
        X = np.empty((img_count, self.height, self.width, 1))
        y = np.empty((img_count, self.height, self.width, 1))
        num_masks_array = np.zeros(img_count)

        # Generate data
        for idx, (k, v) in enumerate(data_dict.items()):

            img_path = '../data/train/' + k + '.dcm'
            im = np.array(pydicom.dcmread(img_path).pixel_array, dtype=float)
            im = (im - np.mean(im)) / np.std(im)
            im = cv2.resize(im, (self.height, self.width))

            X[idx, :, :, 0] = im

            img_name = os.path.splitext(img_path)[0].split("/")[-1]

            # num_masks = self.rle_df[self.rle_df.index == img_name].values()
            num_masks = data_dict[k]

            num_masks_array[idx] = len(num_masks)


            if len(num_masks) > 1:
                y_temp = np.zeros((self.height, self.width))
                for msk_idx in range(len(num_masks)):
                    rle_string = self.rle_df[self.rle_df.index == img_name].values[msk_idx][0]
                    y_temp += self.rle2mask(rle_string)

            elif (num_masks == 0) and (len(self.rle_df[self.rle_df.index == img_name].values) > 0):

                rle_string = self.rle_df[self.rle_df.index == img_name].values[0][0]

                if rle_string == " -1" or rle_string == "-1":
                    y_temp = np.zeros((self.height, self.width))
                    num_masks_array[idx] = 0
                else:
                    y_temp = self.rle2mask(rle_string)

            else:
                y_temp = np.zeros((self.height, self.width))
                num_masks_array[idx] = 0

            y_temp[y_temp > 1] = 1.0
            y[idx, :, :, 0] = y_temp

        return X, y

if __name__ == "__main__":

    training_data = DataGenerator(img_path='../data/train/*.dcm', rle_csv="../data/train-rle.csv", validation=False, batch_size=64,shuffle=False, train_class_one=True)
    images, masks = training_data.__getitem__(1)
