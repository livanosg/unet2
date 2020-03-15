import sys

import tensorflow as tf
eps = 1e-7


def weighted_crossentropy(y_true, y_pred):
    """weighted cross_entropy
    :var y_true onehot labels for n classes
    :var y_pred Softmax output     """

    class_frequencies = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
    weights = tf.math.sqrt(tf.math.divide(tf.reduce_sum(class_frequencies, axis=0), class_frequencies + 1))
    weights_map = weights * y_true
    log = tf.negative(tf.math.log(y_pred))
    wce = tf.math.multiply_no_nan(log, weights_map)
    return tf.reduce_mean(wce)


def weighted_log_dice_loss(y_true, y_pred):
    """ Weighted log-Dice loss
        :var y_true onehot labels for n classes
        :var y_pred Softmax output """
    class_frequencies = tf.reduce_sum(y_true, axis=[1, 2])
    weights = tf.pow(tf.math.divide(1., tf.add(class_frequencies, 1)), 2)
    numerator = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[1, 2]) + 1
    denominator = tf.reduce_sum(tf.math.add(y_true, y_pred), axis=[1, 2]) + 1
    dice = tf.math.multiply(2., tf.math.divide(tf.reduce_sum(tf.math.multiply(weights, numerator), axis=0),
                                               tf.reduce_sum(tf.math.multiply(weights, denominator), axis=0)))
    dice = tf.math.pow(tf.negative(tf.math.log(dice)), 0.3)
    return tf.reduce_mean(dice)


def custom_loss(y_true, y_pred):
    wce = weighted_crossentropy(y_true=y_true, y_pred=y_pred)
    dice = weighted_log_dice_loss(y_true=y_true, y_pred=y_pred)
    loss = tf.math.multiply(0.8, dice) + tf.math.multiply(0.2, wce)
    return loss
