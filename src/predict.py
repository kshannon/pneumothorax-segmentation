# run py script with model path + name as 1st arg e.g. python predict.py 'path/model.h5'
# 2nd python arg is the test dir path

#TODO break apart prediction masks with multiple masks into sepereate mask arrays before translating 
# them into the CSV. this should be done ether before or after RLE encoding

from keras.models import load_model
from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss
import numpy as np
import sys
import os
import csv
import glob
import pydicom
from tqdm import tqdm


custom_objects = {"dice_coef":dice_coef,"dice_coef_loss":dice_coef_loss,"soft_dice_coef":soft_dice_coef}
model = load_model(sys.argv[1], custom_objects=custom_objects)
test_dir = sys.argv[2]


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


with open('../submissions/submission.csv', 'a+', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ImageId','EncodedPixels'])

    for filename in tqdm(os.listdir('../data/test/')): 

        img = np.array(pydicom.dcmread('../data/test/' + filename).pixel_array, dtype=float)
        mean = img.mean()
        std = img.std()
        
        #perfmorming standardization here on the array by sub the mean and dividing the s.d.
        standardized_array = np.divide(np.subtract(img,mean),std)
        
        expanded_array = standardized_array[np.newaxis, ..., np.newaxis]
        msk = model.predict(expanded_array)
        
        if not np.any(msk) == True:
            writer.writerow([filename, '-1']) #case for no pnuemothorax found
        else:
            rle = mask2rle(msk, 1024, 1024)
            writer.writerow([filename, rle]) 



 




