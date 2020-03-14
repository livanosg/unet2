import numpy as np
from tensorflow.keras.backend import epsilon
import tensorflow as tf


def weighted_crossentropy(predictions, labels):
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2], keepdims=True)
        weights = tf.math.sqrt(tf.math.divide(tf.reduce_sum(class_frequencies), class_frequencies + 1))
        wce = tf.math.pow(tf.negative(tf.math.multiply(labels, tf.math.log(tf.clip_by_value(predictions, 1e-7, 1 - 1e-7)))), 0.3)
        wce = tf.math.multiply(weights, wce)
        return tf.reduce_mean(wce)


def weighted_log_dice_loss(predictions, labels):
    """Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso"""

    with tf.name_scope('Generalized_Dice_Loss'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2])
        weights = tf.math.divide(1., tf.pow(tf.add(class_frequencies, 1), 2))
        numerator = tf.math.multiply(2., tf.math.add(tf.reduce_sum(tf.math.multiply(labels, predictions), axis=[0, 1, 2]), 1))
        # numerator = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(labels, predictions) + 1, axis=(0, 1, 2)), weights)) + epsilon()
        denominator = tf.math.add(tf.reduce_sum(tf.math.add(labels, predictions), axis=[0, 1, 2]), 1)
        dice = tf.math.divide(tf.math.multiply(weights, numerator), tf.math.multiply(weights, denominator))
        # denominator = tf.reduce_sum(tf.multiply(tf.reduce_sum(labels + predictions, axis=(0, 1, 2)), weights))
    return tf.reduce_mean(tf.math.pow(tf.negative(tf.math.log(dice)), 0.3))


def custom_loss(y_pred, y_true):        # WCE
        class_frequencies_1 = tf.reduce_sum(y_true, axis=[0, 1, 2], keepdims=True)
        weights = tf.math.sqrt(tf.math.divide(tf.reduce_sum(class_frequencies_1), class_frequencies_1 + 1))
        wce = tf.math.pow(tf.negative(tf.math.multiply(y_true, tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)))), 0.3)
        wce = tf.math.multiply(weights, wce)
        wce = tf.reduce_mean(wce)

        class_frequencies_2 = tf.reduce_sum(y_true, axis=[0, 1, 2])
        weights = tf.math.divide(1., tf.pow(tf.add(class_frequencies_2, 1), 2))
        numerator = tf.math.multiply(2., tf.math.add(tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0, 1, 2]), 1))
        denominator = tf.math.add(tf.reduce_sum(tf.math.add(y_true, y_pred), axis=[0, 1, 2]), 1)
        dice = tf.math.divide(tf.math.multiply(weights, numerator), tf.math.multiply(weights, denominator))
        dice = tf.reduce_mean(tf.math.pow(tf.negative(tf.math.log(dice)), 0.3))
        loss = tf.math.multiply(0.8, dice) + tf.math.multiply(0.2, wce)
        return loss
