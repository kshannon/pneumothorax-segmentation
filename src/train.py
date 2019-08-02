import tensorflow as tf  # conda install -c anaconda tensorflow

import keras as K

from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss

from dataloader import DataGenerator

num_epochs = 100
learningrate = 5e-3
height = 256
width = 256
batch_size = 64

training_data = DataGenerator(img_path="../data/train/*.dcm",
                              rle_csv="../data/train-rle.csv", validation=False,
                              batch_size=batch_size,
                              height=height,
                              width=width,
                              shuffle=True,
                              train_class_one=True)

validation_data = DataGenerator(img_path="../data/train/*.dcm",
                              rle_csv="../data/train-rle.csv", validation=True,
                              batch_size=batch_size,
                              height=height,
                              width=width,
                              shuffle=False,
                              train_class_one=True)

model = unet_model()

opt = K.optimizers.Adam(lr=learningrate)
model.compile(optimizer=opt,loss=dice_coef_loss,
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
              epochs=num_epochs,
              validation_data=validation_data,
              verbose=1, shuffle=True,
              callbacks=[model_checkpoint, tensorboard_checkpoint])
