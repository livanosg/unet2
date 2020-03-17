import tensorflow as tf


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img):
    img = tf.expand_dims(img, 0)
    blur = _gaussian_kernel(3, 3, 1, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    img = tf.squeeze(img, 0)
    return img


def augmentations_fn(dcm, gt_img):
    dcm_ch = tf.shape(dcm)[-1]
    stacked = tf.concat([dcm, tf.cast(gt_img, dcm.dtype)], -1)
    stacked = tf.image.random_flip_left_right(stacked)
    stacked = tf.image.random_flip_up_down(stacked)
    dcm, gt_img = stacked[..., :dcm_ch], stacked[..., dcm_ch:]
    dcm = tf.cond(tf.math.less_equal(tf.random.uniform([1]), 0.5), lambda: apply_blur(dcm), lambda: dcm)
    dcm = tf.image.random_contrast(dcm, 0.6, 1.3)
    return dcm, gt_img
