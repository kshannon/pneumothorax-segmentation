# import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import pickle as pickle
print(pickle.HIGHEST_PROTOCOL)
import bz2
from tqdm import tqdm
import os
import pydicom
import glob
from PIL import Image
import gdcm
import PIL
import matplotlib.pyplot as plt

#path defs
train_path = '../data/dicom-images-train/'
test_path = '../data/dicom-images-test/'
train_dicom_names = '../data/train-dicom-names.csv'
test_dicom_names = '../data/test-dicom-names.csv'
rle_csv = '../data/train-rle.csv'

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;
    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;
    return " ".join(rle)


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    return mask.reshape(width, height)


def generate_data(image_dim=(1024,1024,1)):
    # import RLE csv file
    df = pd.read_csv(rle_csv, index_col='ImageId') #TODO take extra nagging space out of encoded pixels
    df.head()

    train_glob = train_path + '*/*/*.dcm'
    test_glob = test_path + '*/*/*.dcm'
    train_fns = sorted(glob.glob(train_glob))[:200] # change here to limit data
    test_fns = sorted(glob.glob(test_glob))[:200] # change here to limit data
    df = pd.read_csv('../data/train-rle.csv', index_col='ImageId')

    height = image_dim[0]
    width = image_dim[1]
    chan = image_dim[2]

    # Get train images and masks
    X_train = np.zeros((len(train_fns), height, width, chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_fns), height, width, 1), dtype=np.bool)
    print('Getting train images and masks ... ')
    sys.stdout.flush()

    for n, _id in tqdm(enumerate(train_fns), total=len(train_fns)):
        dataset = pydicom.read_file(_id)
        X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
        try:
            if '-1' in df.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
                Y_train[n] = np.zeros((1024, 1024, 1))
            else:
                if type(df.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
                    Y_train[n] = np.expand_dims(rle2mask(df.loc[_id.split('/')[-1][:-4],' EncodedPixels'], 1024, 1024), axis=2)
                else:
                    Y_train[n] = np.zeros((1024, 1024, 1))
                    for x in df.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
                        Y_train[n] =  Y_train[n] + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
        except KeyError:
            print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
            Y_train[n] = np.zeros((1024, 1024, 1)) # Assume missing masks are empty masks.

    print('Done!')

    yfile = bz2.BZ2File("../data/y_train.pkl", 'w')
    pickle.dump(Y_train, yfile, protocol=0)
    xfile = bz2.BZ2File("../data/x_train.pkl", 'w')
    pickle.dump(X_train, xfile, protocol=0)
    print('Saved ML ready data objects to disk!')

if __name__ == '__main__':
    generate_data()


    with open("../data/x_train.pkl", "rb") as infile:
        x_train = pickle.load(infile, encoding='latin1')

    with open("../data/y_train.pkl", "rb") as infile:
        y_train = pickle.load(infile, encoding='latin1')
    # try:
    #     with open(r"../data/x_train", "rb") as infile:
    #         x_train = pickle.load(infile, encoding='bytes')

    #     with open(r"../data/y_train", "rb") as infile:
    #         y_train = pickle.load(infile, encoding='bytes')
    # except:
    #     generate_data()

    print(type(x_train))
    print(type(y_train))


# TODO: check for pickle files, if exists import, if not run script
# 
# TODO: add in hooks for tf.data and keras dataloader from pickle scripts
# 
# TODO: u0net baseline
# 
# TODO: output submission script (import as module) 