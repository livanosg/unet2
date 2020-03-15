import tensorflow as tf
import tensorflow.keras.backend as K

eps = 1e-7


def weighted_crossentropy(y_true, y_pred):
    """weighted cross_entropy
    :var y_true onehot labels for n classes
    :var y_pred Softmax output     """
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    class_frequencies = tf.reduce_sum(y_true, axis=[0, 1, 2])
    weights = tf.math.sqrt(tf.math.divide(tf.reduce_sum(class_frequencies), class_frequencies + 1))
    weights_map = weights * y_true
    log = tf.negative(tf.math.log(y_pred))
    wce = tf.math.multiply_no_nan(log, weights_map)
    wce = tf.math.pow(wce, 0.3)
    return tf.reduce_mean(wce)


def weighted_log_dice_loss(y_true, y_pred):
    """ Weighted log-Dice loss
        :var y_true onehot labels for n classes
        :var y_pred Softmax output """
    class_frequencies = tf.reduce_sum(y_true, axis=[0, 1, 2])
    weights = tf.math.divide(tf.reduce_sum(class_frequencies) - class_frequencies, class_frequencies + 1)
    weights_map = weights * y_true
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights_map_f = K.flatten(weights_map)

    numerator = y_true_f * y_pred_f
    denominator = tf.math.pow(y_true_f, 2.) + tf.math.pow(y_pred_f, 2.)
    dice = (2 * (weights_map_f * numerator + 1)) / (weights_map_f * denominator + 1)
    dice = tf.math.pow(tf.negative(tf.math.log(dice)), 0.3)
    return tf.reduce_mean(dice)


def custom_loss(y_true, y_pred):
    wce = weighted_crossentropy(y_true=y_true, y_pred=y_pred)
    dice = weighted_log_dice_loss(y_true=y_true, y_pred=y_pred)
    loss = tf.math.multiply(0.8, dice) + tf.math.multiply(0.2, wce)
    return loss
