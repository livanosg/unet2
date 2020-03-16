import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa
from data_generators import data_gen, get_paths


def input_fn(mode, args):
    """ input_fn for tf.estimator for TRAIN, EVAL and PREDICT modes.
    Inputs
    mode -> one of tf.estimator modes defined from tf.estimator.ModeKeys
    params -> arguments passed to data_generator and batch size"""
    logs = tf.get_logger()
    logs.warning(' Setting up {} dataset iterator...'.format(mode))
    if mode in ('train', 'testing', 'eval'):
        if mode in ('train', 'testing'):
            data_set = tf.data.Dataset.from_generator(generator=lambda: data_gen('train', args),
                                                      output_types=(tf.float32, tf.int32))
        else:
            data_set = tf.data.Dataset.from_generator(generator=lambda: data_gen('eval', args),
                                                      output_types=(tf.float32, tf.int32))
        data_set = data_set.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=args.classes)))
        data_set = data_set.map(lambda x, y: (tf.cast(x, tf.float32), y))
        data_set = data_set.map(lambda x, y: (tf.expand_dims(x, -1), y))
        if mode in ('train', 'testing'):
            data_set = data_set.batch(args.batch_size)
        if mode == 'eval':
            data_set = data_set.batch(1)
        data_set = data_set.repeat()
    else:
        data_set = tf.data.Dataset.from_generator(generator=lambda: data_gen(mode, args),
                                                  output_types=(tf.float32, tf.string))
        data_set = data_set.map(lambda x, y: (tf.expand_dims(x, -1), y))
        data_set = data_set.batch(args.batch_size)
    data_set = data_set.prefetch(buffer_size=-1)
    return data_set


def input_dcm(mode, args):
    logs = tf.get_logger()
    logs.warning(' Setting up {} dataset iterator...'.format(mode))
    if mode in ('train', 'testing', 'eval'):
        if mode in ('train', 'testing'):
            paths = get_paths('train', args.modality)
            np.random.shuffle(paths)
        else:
            paths = get_paths('eval', args.modality)
        dcm_paths, grd_paths = zip(*paths)
        dcm_paths = list(dcm_paths)
        grd_paths = list(grd_paths)
        dcm_data_set = tf.data.Dataset.from_tensor_slices(dcm_paths)
        dcm_data_set = dcm_data_set.map(lambda x: tfio.image.decode_dicom_image(tf.io.read_file(filename=x), color_dim=True, scale='preserve', dtype=tf.float32))
        dcm_data_set = dcm_data_set.map(lambda x: tf.image.per_image_standardization(x))
        dcm_data_set = dcm_data_set.map(lambda x: tf.image.random_contrast(x, 0.5, 1.5))
        dcm_data_set = dcm_data_set.map(lambda x: tf.image.random_brightness(x, max_delta=0.5))
        dcm_data_set = dcm_data_set.map(lambda x: apply_blur(x))

        png_data_set = tf.data.Dataset.from_tensor_slices(grd_paths)
        png_data_set = png_data_set.map(lambda x: tf.io.decode_png(tf.io.read_file(filename=x), dtype=tf.uint8))
        png_data_set = png_data_set.map(lambda x: label_formating(x))
        dataset = tf.data.Dataset.zip((dcm_data_set, png_data_set))
        dataset = dataset.map(lambda img, gt_img: augmentations(img=img, gt_img=gt_img))
        dataset = dataset.map(lambda x, y:(x,  tf.one_hot(tf.cast(tf.squeeze(y, -1), tf.int32), depth=args.classes)))

        if mode in ('train', 'testing'):
            dataset = dataset.batch(args.batch_size)
        if mode == 'eval':
            dataset = dataset.batch(1)
        dataset = dataset.repeat()
    else:
        paths = get_paths(mode, args.modality)
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(
            lambda x: tfio.image.decode_dicom_image(x, color_dim=True, scale='auto', dtype=tf.float32))
        dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(buffer_size=-1)
    return dataset


def label_formating(gt_img):
    return tf.where(tf.logical_or(tf.equal(gt_img, 63), tf.equal(gt_img, 255)), tf.ones_like(gt_img), tf.zeros_like(gt_img))


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img):
    blur = _gaussian_kernel(3, 2, 1, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def augmentations(img, gt_img):
    img_dims, img_ch = tf.shape(img)[1:3], tf.shape(img)[-1]
    stacked = tf.concat([img, tf.cast(gt_img, img.dtype)], -1)
    stacked = tf.image.random_flip_left_right(stacked)
    stacked = tf.image.random_flip_up_down(stacked)
    a = tf.random.uniform([2], img_dims[0]//4, img_dims[0]//4, tf.int32)
    stacked = tf.image.crop_to_bounding_box(stacked,a[0], a[1])
    return tfa.image.rotate(stacked[..., :img_ch], angles=a), tfa.image.rotate(stacked[..., img_ch:], angles=a)


if __name__ == '__main__':
    import cv2


    class args:
        def __init__(self):
            self.modality = 'CT'
            self.batch_size = 1
            self.classes = 2

    cv2.namedWindow('TEST', cv2.WINDOW_FREERATIO)
    args = args()
    for dcm, png in input_dcm('train', args):
        dcm = dcm.numpy()
        dcm = np.squeeze(dcm, 0)
        png = png.numpy()
        png = np.squeeze(png, 0)
        png = np.expand_dims(png[:, :, 1],-1)

        stack = np.hstack((dcm/np.max(dcm), png/np.max(png)))

        cv2.imshow('TEST', stack)
        cv2.waitKey()
