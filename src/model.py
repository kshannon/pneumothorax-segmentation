#!/usr/bin/env python
#
# -*- coding: utf-8 -*-

import tensorflow as tf  # conda install -c anaconda tensorflow

import keras as K

def dice_coef(target, prediction, axis=(1, 2), smooth=0.01):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    prediction = K.backend.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)

def soft_dice_coef(target, prediction, axis=(1, 2), smooth=0.01):
    """
    Sorenson (Soft) Dice  - Don't round the predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)

def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.0):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def unet_model():
    """
    U-Net Model
    """

    num_chan_in = 1
    num_chan_out = 1
    fms=32
    dropout=0.2
    concat_axis=-1

    inputs = K.layers.Input([None, None, num_chan_in], name="CXR")

    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same",
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    encodeA = K.layers.Conv2D(name="encodeAa", filters=fms, **params)(inputs)
    encodeA = K.layers.Conv2D(name="encodeAb", filters=fms, **params)(encodeA)
    poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

    encodeB = K.layers.Conv2D(name="encodeBa", filters=fms*2, **params)(poolA)
    encodeB = K.layers.Conv2D(
        name="encodeBb", filters=fms*2, **params)(encodeB)
    poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

    encodeC = K.layers.Conv2D(name="encodeCa", filters=fms*4, **params)(poolB)
    encodeC = K.layers.SpatialDropout2D(dropout)(encodeC)
    encodeC = K.layers.Conv2D(
        name="encodeCb", filters=fms*4, **params)(encodeC)

    poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

    encodeD = K.layers.Conv2D(name="encodeDa", filters=fms*8, **params)(poolC)
    encodeD = K.layers.SpatialDropout2D(dropout)(encodeD)
    encodeD = K.layers.Conv2D(
        name="encodeDb", filters=fms*8, **params)(encodeD)

    poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

    encodeE = K.layers.Conv2D(name="encodeEa", filters=fms*16, **params)(poolD)
    encodeE = K.layers.Conv2D(
        name="encodeEb", filters=fms*16, **params)(encodeE)

    up = K.layers.UpSampling2D(name="upE", size=(2, 2),
                                   interpolation="bilinear")(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=concat_axis, name="concatD")

    decodeC = K.layers.Conv2D(
        name="decodeCa", filters=fms*8, **params)(concatD)
    decodeC = K.layers.Conv2D(
        name="decodeCb", filters=fms*8, **params)(decodeC)

    up = K.layers.UpSampling2D(name="upC", size=(2, 2),
                                   interpolation="bilinear")(decodeC)

    concatC = K.layers.concatenate(
        [up, encodeC], axis=concat_axis, name="concatC")

    decodeB = K.layers.Conv2D(
        name="decodeBa", filters=fms*4, **params)(concatC)
    decodeB = K.layers.Conv2D(
        name="decodeBb", filters=fms*4, **params)(decodeB)

    up = K.layers.UpSampling2D(name="upB", size=(2, 2),
                                   interpolation="bilinear")(decodeB)

    concatB = K.layers.concatenate(
        [up, encodeB], axis=concat_axis, name="concatB")

    decodeA = K.layers.Conv2D(
        name="decodeAa", filters=fms*2, **params)(concatB)
    decodeA = K.layers.Conv2D(
        name="decodeAb", filters=fms*2, **params)(decodeA)

    up = K.layers.UpSampling2D(name="upA", size=(2, 2),
                                   interpolation="bilinear")(decodeA)

    concatA = K.layers.concatenate(
        [up, encodeA], axis=concat_axis, name="concatA")

    convOut = K.layers.Conv2D(name="convOuta", filters=fms, **params)(concatA)
    convOut = K.layers.Conv2D(name="convOutb", filters=fms, **params)(convOut)

    prediction = K.layers.Conv2D(name="PredictionMask",
                                 filters=num_chan_out, kernel_size=(1, 1),
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    return model
