import os
from os import environ
from glob import glob
from math import ceil
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import Conv2D, Softmax

from callbacks import InputOutputResults
from config import paths
from data_generators import data_gen
from input_fns import train_eval_input_fn
from loss_fn import custom_loss
from lr_schedules import lr_schedule
from metrics import dice_micro, dice_macro, dice_weighted
from archit import down_block, double_conv, up_block


def train_fn(args):
    # Distribution Strategy
    environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    environ['TF_XLA_FLAGS'] = 'none'
    # environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + root_dir
    # TODO Implement on multi-nodes SLURM
    tf.config.set_soft_device_placement(enabled=True)

    if args.mode == 'test':
        train_size = 10
        eval_size = 5
    else:
        train_size = len(list(data_gen(dataset=tf.estimator.ModeKeys.TRAIN, args=args, only_paths=True)))
        eval_size = len(list(data_gen(dataset=tf.estimator.ModeKeys.EVAL, args=args, only_paths=True)))
    train_dataset = train_eval_input_fn(mode='train', args=args)
    eval_dataset = train_eval_input_fn(mode='eval', args=args)
    print('Train data size: {}, Eval_data size: {}'.format(train_size, eval_size))

    args.epoch_steps = ceil(train_size / args.batch_size)
    args.eval_steps = ceil(eval_size / args.batch_size)
    learning_rate = lr_schedule(args=args)
    optimizer = Adam(learning_rate=args.lr)
    metrics = [dice_micro, dice_macro, dice_weighted]


    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('/device:CPU:0')

    if args.load_model:
        model_path = paths['save'] + '/' + args.load_model
        assert os.path.exists(model_path)
        load_model_path = sorted(glob(model_path + '/model_saves/*'))[-1]

    if not args.load_model or not args.resume:
        print('New training session')
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
        model_saves = model_path + '/model_saves'
        os.makedirs(model_saves, exist_ok=True)

    print('Model will be saved at : {}'.format(model_path))
    if args.load_model:
        custom_objects = {'custom_loss': custom_loss, 'dice_micro': dice_micro,
                          'dice_macro': dice_macro, 'dice_weighted': dice_weighted, 'tensorboard_callback': tensorboard_callback, 'images': images, 'learning_rate': learning_rate}
        with tf.keras.utils.custom_object_scope(custom_objects):
            print('Model will be loaded from : {}'.format(load_model_path))
            model = tf.keras.models.load_model(load_model_path)                                 # with compile=True ==> RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)              # with compile=False ==> Same
    if not args.load_model or not args.resume:
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)
