import tensorflow as tf


def format_gray_image(tensor):
    """Gets a tf.tensor of dims bxHxW, bxHxWxc
    and converts it to bxHxWxc."""
    shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    assert len(shape) >= 3
    assert shape[-1] in range(1, 5)
    if len(shape) == 3:
        tensor = tf.expand_dims(tensor, -1)

    if shape[-1] > 1:
        tensor = tf.math.argmax(tensor, axis=-1)
        tensor = tf.expand_dims(tensor, -1)

    gray_values = tf.cast(tf.math.maximum(1, shape[-1] - 1), tf.float32)

    max_val = tf.math.reduce_max(tensor, axis=[0, 1, 2])
    min_val = tf.math.reduce_min(tensor, axis=[0, 1, 2])
    if tf.math.equal(max_val, min_val):

        image = tf.math.divide(tf.cast(tensor, tf.float32), gray_values)
    else:
        numerator = tf.math.subtract(tensor, min_val)
        denom = tf.math.subtract(max_val, min_val)
        image = tf.math.divide(numerator, denom)
        image = tf.math.divide(tf.cast(image, tf.float32), gray_values)
    return tf.cast(tf.math.multiply(image, 255), dtype=tf.uint8)
