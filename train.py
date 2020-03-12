import os
from os import environ
from glob import glob
from math import ceil
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from callbacks import PrintLR, InputOutputResults
from config import paths
from data_generators import data_gen
from input_fns import train_eval_input_fn
from loss_fn import custom_loss
from metrics import dice_micro, dice_macro, dice_weighted
from archit import unet


# noinspection PyUnboundLocalVariable
def train_fn(args):
    # Distribution Strategy
    environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # TODO Implement on multi-nodes SLURM
    tf.config.set_soft_device_placement(enabled=True)  # Choose CPU if no GPu
    # session_config = tf.config.allow_growth = True  # Allow full memory usage of GPU.\

    if args.load_model and args.resume:
        model_path = paths['save'] + '/' + args.load_model
    else:
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
    print('Model saved at: {}'.format(model_path))
    logs_path = model_path + '/logs/'

    if args.mode == 'test':
        train_size = 5
        eval_size = 2
    else:
        train_size = len(list(data_gen(dataset=tf.estimator.ModeKeys.TRAIN, args=args, only_paths=True)))
        eval_size = len(list(data_gen(dataset=tf.estimator.ModeKeys.EVAL, args=args, only_paths=True)))

    print('Train data size: {}, Eval_data size: {}'.format(train_size, eval_size))
    train_dataset = train_eval_input_fn(mode='train', args=args)
    eval_dataset = train_eval_input_fn(mode='eval', args=args)

    steps_per_epoch = ceil(train_size / args.batch_size)
    print('Epochs: {}, Steps per epoch: {}.'.format(args.epochs, steps_per_epoch))

    # decay_steps = ceil(args.epochs * steps_per_epoch / (args.decays_per_train + 1))
    # learning_rate = schedules.ExponentialDecay(initial_learning_rate=args.lr, decay_steps=decay_steps,
    #                                            decay_rate=args.decay_rate)
    # learning_rate = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=args.lr,
    #                                                     maximal_learning_rate=5 * args.lr,
    #                                                     step_size=5 * steps_per_epoch)
    learning_rate = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=args.lr,
                                                                  maximal_learning_rate=5 * args.lr,
                                                                  step_size=3 * steps_per_epoch)
    optimizer = Adam(learning_rate=learning_rate)
    #
    # def get_model(dropout=0.5, batch_norm=True):
    #     # x = tf.keras.layers.Input(shape=(None, None, 1), name='inputs')
    #     # Wrap what's loaded to a KerasLayer
    #     # keras_layer = hub.KerasLayer(loaded, trainable=True)
    #     # model = tf.keras.Model(keras_layer.input, keras_layer.output)
    #     return unet(dropout=dropout, batch_norm=batch_norm)

    # Define metrics
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # model = unet(dropout=0.5, batch_norm=True)  # loading
        if args.load_model:
            load_path = paths['save'] + '/' + args.load_model
            model_path = sorted(glob(load_path + '/model'))
            assert len(model_path) != 0
            model_path = model_path[-1]
            print('Loading model from {}'.format(model_path))
            model = tf.keras.models.load_model(model_path,
                                               custom_objects={'custom_loss': custom_loss, 'dice_micro': dice_micro,
                                                               'dice_macro': dice_macro,
                                                               'dice_weighted': dice_weighted},compile=False)
            model.compile(optimizer=optimizer, loss=custom_loss, metrics=[dice_micro, dice_macro, dice_weighted])
            if not args.resume:
                print('Weights will be loaded from : {}'.format(model_path))
                model.compile(optimizer=optimizer, loss=custom_loss, metrics=[dice_micro, dice_macro, dice_weighted])

        else:
            model = unet(dropout=0.5, batch_norm=True)
            model.compile(optimizer=optimizer, loss=custom_loss, metrics=[dice_micro, dice_macro, dice_weighted])
        print('Loaded.')
    # Define  callbacks.
        lr_cb = PrintLR(model=model)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, write_images=True)
        best_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path + '/model', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        images = InputOutputResults(model=model, logdir=logs_path, train_data=train_dataset, eval_data=eval_dataset, args=args)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model.fit(x=train_dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, validation_data=eval_dataset, validation_steps=eval_size, callbacks=[tensorboard_callback, best_checkpoint, lr_cb, images, early_stop])
