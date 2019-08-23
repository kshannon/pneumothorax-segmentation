# run py script with model path + name as 1st arg e.g. python predict.py 'path/model.h5'
# set plot to True to see prediction masks and test images

#TODO break apart prediction masks with multiple masks into sepereate mask arrays before translating 
# them into the CSV. this should be done ether before or after RLE encoding

from keras.models import load_model
from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss
import numpy as np
import sys
import os
import csv
import glob
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm


custom_objects = {"dice_coef":dice_coef,"dice_coef_loss":dice_coef_loss,"soft_dice_coef":soft_dice_coef}
model = load_model(sys.argv[1], custom_objects=custom_objects)
plot = sys.argv[2]


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
                if currentColor == 1: #was 255...
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


def rle2mask(rle):
    mask= np.zeros(1024 * 1024)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255 #255 for white pixel, 0 is a black pixel
        current_position += lengths[index]
    
    return mask.reshape(1024,1024).T  # Because mask is rotated


with open('../submissions/submission.csv', 'a+', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ImageId','EncodedPixels'])

    for filename in tqdm(os.listdir('../data/test/')): 

        img = np.array(pydicom.dcmread('../data/test/' + filename).pixel_array, dtype=float).T
        mean = img.mean()
        std = img.std()
        
        #perfmorming standardization here on the array by sub the mean and dividing the s.d.
        standardized_array = np.divide(np.subtract(img,mean),std)
        
        expanded_array = standardized_array[np.newaxis, ..., np.newaxis]
        msk = model.predict(expanded_array)
        msk = np.squeeze(np.round(msk)) #remove axis w/ dims of 1, round mask for given probbabilties to [0,1]
        
        if not np.any(msk) == True:
            writer.writerow([filename, '-1']) #case for no pnuemothorax found
            print('none found...')
        else:
            rle = mask2rle(msk, 1024, 1024)
            writer.writerow([filename, rle])

        # if plot != False:
        #     plt.imshow(img.T, cmap=plt.cm.bone)
        #     plt.show()
        #     plt.imshow(rle2mask(msk), cmap=plt.cm.bone) #TODO some issue with .split(), for now use model validation notebook
        #     plt.show()



 




