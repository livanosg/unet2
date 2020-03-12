import tensorflow as tf

from data_generators import data_gen, test_gen


def train_eval_input_fn(mode, args):
    """ input_fn for tf.estimator for TRAIN, EVAL and PREDICT modes.
    Inputs
    mode -> one of tf.estimator modes defined from tf.estimator.ModeKeys
    params -> arguments passed to data_generator and batch size"""
    logs = tf.get_logger()
    logs.warning(' Setting up {} dataset iterator...'.format(mode))
    # Eval and predict options
    if mode in (tf.estimator.ModeKeys.EVAL):
        args.augm_prob = 0.
        args.augm_set = None
        args.shuffle = False
        args.batch_size = 1

    with tf.name_scope('Feeding_Mechanism'):
        # Don't declare generator to a variable or else Dataset.from_generator cannot instantiate the generator
        data_set = tf.data.Dataset.from_generator(generator=lambda: data_gen(mode, args),
                                                  output_types=(tf.float32, tf.int32),
                                                  output_shapes=(tf.TensorShape([None,None]), tf.TensorShape([None, None])))
        data_set = data_set.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=args.classes)))
        data_set = data_set.map(lambda x, y: (tf.cast(x, tf.float32), y))
        data_set = data_set.map(lambda x, y: (tf.expand_dims(x, -1), y))
    if mode == 'train':
        data_set = data_set.batch(args.batch_size)
    if mode == 'eval':
        data_set = data_set.batch(1)
    data_set = data_set.prefetch(buffer_size=-1)
    return data_set


def pred_input_fn(params):
    data_set = tf.data.Dataset.from_generator(generator=lambda: test_gen(params=params),
                                              output_types=(tf.float32, tf.string),
                                              output_shapes=(tf.TensorShape([None, None]), tf.TensorShape(None)))
    data_set = data_set.map(lambda x, y: {'image': tf.expand_dims(x, -1), 'path': tf.cast(y, tf.string)})
    data_set = data_set.batch(params['batch_size'])
    data_set = data_set.prefetch(buffer_size=-1)
    return data_set
