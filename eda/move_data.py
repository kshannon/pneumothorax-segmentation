import shutil
import os
import glob2
from tqdm import tqdm

train_path = '../data/train/'
test_path = '../data/test/'

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

test_num = 0
for filename in tqdm(glob2.glob('../data/dicom-images-test/**/*.dcm')):
    fname = str(filename).split('/')[-1]
    shutil.copy(str(filename), os.path.join(test_path, fname))
    test_num += 1

print('Moved ' + str(test_num) + ' files!' )


train_num = 0
for filename in tqdm(glob2.glob('../data/dicom-images-train/**/*.dcm')):
    fname = str(filename).split('/')[-1]
    shutil.copy(str(filename), os.path.join(train_path, fname))
    train_num += 1

print('Moved ' + str(train_num) + ' files!' )