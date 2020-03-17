import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from augmentations import augmentations_fn
from config import root_dir
from paths import get_paths
import sys

print(sys.version)

os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + root_dir


def input_dcm(mode, args):
    logs = tf.get_logger()
    logs.info(' Setting up {} dataset iterator...'.format(mode))
    read_dcm = lambda x: tfio.image.decode_dicom_image(tf.io.read_file(filename=x), scale='preserve', dtype=tf.float32)
    read_png = lambda x: tf.io.decode_png(tf.io.read_file(filename=x), dtype=tf.uint8)
    if mode in ('train', 'testing', 'eval'):
        if mode in ('train', 'testing'):
            paths = get_paths('train', args.modality)
            np.random.shuffle(paths)
        else:
            paths = get_paths('eval', args.modality)
            np.random.shuffle(paths)
        dcm_paths, grd_paths = zip(*paths)
        dcm_paths = list(dcm_paths)
        grd_paths = list(grd_paths)
        dcm_data_set = tf.data.Dataset.from_tensor_slices(dcm_paths)
        png_data_set = tf.data.Dataset.from_tensor_slices(grd_paths)
        dataset = tf.data.Dataset.zip((dcm_data_set, png_data_set))
        dataset = dataset.map(lambda x, y: (tf.squeeze(read_dcm(x), axis=0), read_png(y)))
        if mode in ('train', 'testing'):
            dataset = dataset.map(augmentations_fn)
            if args.modality in ('CT', 'ALL'):
                new_shape = 512
            else:
                new_shape = 320

            dataset = dataset.map(
                lambda x, y: (tf.image.pad_to_bounding_box(x, offset_height=(new_shape - tf.shape(x)[0]) // 2,
                                                           offset_width=(new_shape - tf.shape(x)[1]) // 2,
                                                           target_height=new_shape, target_width=new_shape),
                              tf.image.pad_to_bounding_box(y, offset_height=(new_shape - tf.shape(y)[0]) // 2,
                                                           offset_width=(new_shape - tf.shape(y)[1]) // 2,
                                                           target_height=new_shape, target_width=new_shape)))

        dataset = dataset.map(lambda x, y: standardize(x, y, args.classes))
        if mode in ('train', 'testing'):
            dataset = dataset.batch(args.batch_size)
        if mode == 'eval':
            dataset = dataset.batch(1)
    else:
        paths = get_paths(mode, args.modality)
        dataset = tf.data.Dataset.from_tensor_slices(paths)

        dataset = dataset.map(read_dcm)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.map(lambda x: tf.image.per_image_standardization(x))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=-1)
    return dataset


def standardize(dcm, label, classes):
    label = tf.where(tf.logical_or(tf.equal(label, 63), tf.equal(label, 255)), tf.ones_like(label),
                     tf.zeros_like(label))
    label = tf.one_hot(tf.cast(tf.squeeze(label, -1), tf.int32), depth=classes)
    dcm = tf.image.per_image_standardization(dcm)
    return dcm, label
