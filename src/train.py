import tensorflow as tf  # conda install -c anaconda tensorflow

import keras as K

from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss

from dataloader import DataGenerator

training_data = DataGenerator(im_path="dicom-images-train/*/*/*.dcm",
                              rle_csv="train-rle.csv", testing=False,
                              batch_size=64,
                              shuffle=True)

validation_data = DataGenerator(im_path="dicom-images-train/*/*/*.dcm",
                              rle_csv="train-rle.csv", testing=True,
                              batch_size=64,
                              shuffle=False)

model = unet_model()

model.compile(optimizer="adam",loss=dice_coef_loss,
              metrics=[dice_coef, soft_dice_coef])


model_filename = "pneumothorax.hdf5"
model_checkpoint = K.callbacks.ModelCheckpoint(model_filename,
                                               verbose=1,
                                               monitor="val_loss",
                                               save_best_only=True)

tensorboard_filename = "tb_logs"
tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename,
            write_graph=True, write_images=True)

model.fit_generator(training_data,
              epochs=30,
              validation_data=validation_data,
              verbose=1, shuffle=True,
              callbacks=[model_checkpoint, tensorboard_checkpoint])
