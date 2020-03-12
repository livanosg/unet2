import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, AcceptableDTypes
from typeguard import typechecked

eps = 1e-8


def dice_micro(y_true, y_pred):
    """Weighted Dice Loss computed from class-map label.
    dice_type: 'micro' 'macro' 'weighted'
    """

    # F1 micro
    true_positives = tf.reduce_sum(tf.math.multiply(y_pred, y_true))  # <== [class_1, class_2]
    false_positives = tf.reduce_sum(tf.math.abs(tf.math.multiply(y_pred, tf.math.subtract(y_true, 1))))
    false_negatives = tf.reduce_sum(tf.math.abs(tf.math.multiply(tf.math.subtract(y_pred, 1), y_true)))
    precision = tf.math.divide_no_nan(true_positives, true_positives + false_positives)
    recall = tf.math.divide_no_nan(true_positives, true_positives + false_negatives)
    mul_value = tf.math.multiply(precision, recall)
    add_value = tf.math.add(precision, recall)
    mean = tf.math.divide_no_nan(mul_value, add_value)
    f1_score = 2 * mean
    return tf.math.reduce_mean(f1_score)


def dice_macro(y_true, y_pred):
    """Weighted Dice Loss computed from class-map label.
    dice_type: 'micro' 'macro' 'weighted'"""
    true_positives = tf.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[0, 1, 2])  # <== [class_1, class_2]
    false_positives = tf.reduce_sum(tf.math.abs(tf.math.multiply(y_pred, tf.math.subtract(y_true, 1))), axis=[0, 1, 2])
    false_negatives = tf.reduce_sum(tf.math.abs(tf.math.multiply(tf.math.subtract(y_pred, 1), y_true)), axis=[0, 1, 2])
    precision = tf.math.divide_no_nan(true_positives, true_positives + false_positives)
    recall = tf.math.divide_no_nan(true_positives, true_positives + false_negatives)
    mul_value = tf.math.multiply(precision, recall)
    add_value = tf.math.add(precision, recall)
    mean = tf.math.divide_no_nan(mul_value, add_value)
    f1_score = 2 * mean
    return tf.math.reduce_mean(f1_score)


def dice_weighted(y_true, y_pred):
    weights_intermediate = tf.reduce_sum(y_true, axis=[0, 1, 2])
    weights = tf.math.divide_no_nan(weights_intermediate, tf.reduce_sum(weights_intermediate))
    true_positives = tf.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[0, 1, 2])  # <== [class_1, class_2]
    false_positives = tf.reduce_sum(tf.math.abs(tf.math.multiply(y_pred, tf.math.subtract(y_true, 1))),
                                    axis=[0, 1, 2])
    false_negatives = tf.reduce_sum(tf.math.abs(tf.math.multiply(tf.math.subtract(y_pred, 1), y_true)),
                                    axis=[0, 1, 2])
    precision = tf.math.divide_no_nan(true_positives, true_positives + false_positives)
    recall = tf.math.divide_no_nan(true_positives, true_positives + false_negatives)
    mul_value = tf.math.multiply(precision, recall)
    add_value = tf.math.add(precision, recall)
    mean = tf.math.divide_no_nan(mul_value, add_value)
    f1_score = 2 * mean
    f1_score = tf.reduce_sum(f1_score * weights)
    return tf.math.reduce_mean(f1_score)
