import pandas as pd
import os
import pydicom
import glob
from PIL import Image
import gdcm
import PIL

#path defs
train_path = '../data/dicom-images-train/'
test_path = '../data/dicom-images-test/'
out_path = '../data/'

# make a list of filepaths+filenames for all  data
train_paths = []
test_paths = []
for idx, data_set_path in enumerate([train_path, test_path]):
    for filename in glob.iglob(data_set_path + '**/*.dcm', recursive=True):
        if idx == 0:
            train_paths.append(filename)
        else:
            test_paths.append(filename)
assert len(train_paths) == 10712
assert len(test_paths) == 1377

# Extract out metadata attributes from dicom class
train_metadata = []
test_metadata = []
for idx, data_set in enumerate([train_paths, test_paths]):
    for dicom in data_set:
        ds = pydicom.dcmread(dicom, force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        data_dict = {} 

        data_dict["modality"] = getattr(ds, "Modality", None) 
        data_dict["conversion_type"] = getattr(ds, "ConversionType", None)
        data_dict["sex"] = getattr(ds, "PatientSex", None)
        data_dict["age"] = getattr(ds, "PatientAge", None)
        data_dict["body_part"] = getattr(ds, "BodyPartExamined", None)
        data_dict["view_pos"] = getattr(ds, "ViewPosition", None)
        data_dict["rows"] = getattr(ds, "Rows", None)
        data_dict["cols"] = getattr(ds, "Columns", None)
        data_dict["pixels"] = getattr(ds, "PixelData", None)
        
        samples = getattr(ds, "SamplesPerPixel", None)
        if samples != None:
            int(samples)
        data_dict["samples_per_pixel"] = samples
        
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing != None:
            a,b = float(spacing[0]),float(spacing[1])
        else:
            a,b = None, None
        data_dict["pixel_spacing_one"] = a
        data_dict["pixel_spacing_two"] = b

        data_dict["dicom_file"] = dicom.split('/')[-1]
        data_dict["dicom_path"] = dicom

        if idx == 0:
            train_metadata.append(data_dict)
        else:
            test_metadata.append(data_dict)

assert len(train_metadata) == 10712
assert len(test_metadata) == 1377
print(len(train_metadata) + ' number of train data processed')
print(len(test_metadata) + ' number of test data processed')

# Create and save/pickle pandas DFs for future use
train_df = pd.DataFrame.from_dict(train_metadata)
test_df = pd.DataFrame.from_dict(test_metadata)
train_df.to_pickle(out_path + "train_metadata_df.pkl")
test_df.to_pickle(out_path + "test_metadata_df.pkl")


