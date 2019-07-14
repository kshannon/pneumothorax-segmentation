# run py script with model path + name as 1st arg e.g. python predict.py 'path/model.h5'
# 2nd python arg is the test dir path

#TODO break apart prediction masks with multiple masks into sepereate mask arrays before translating 
# them into the CSV. this should be done ether before or after RLE encoding

from keras.models import load_model
import numpy as np
import sys
import glob

model = load_model(sys.argv[0])
test_dir = sys.argv[1]


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


with open('../submissions/submission.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ImageId','EncodedPixels'])

    for filename in glob.iglob(test_dir + '**/*.dcm', recursive=True):
        img_id = filename.split('/')[-1]
        msk = model.predict(filename)

        if not np.any(msk) == True:
            writer.writerow(dicom_name, '-1') #case for no pnuemothorax found
        else:
            rle = mask2rle(msk, 1024, 1024)
            writer.writerow(img_id, rle) 




 




