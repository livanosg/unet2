import os
from os import environ
from glob import glob
from math import ceil
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from archit import unet
from config import paths
from data_generators import data_gen
from input_fns import train_eval_input_fn
from loss_fn import custom_loss
from lr_schedules import lr_schedule
from metrics import dice_weighted, dice_micro, dice_macro
from callbacks import ModelCheckpoint


# noinspection PyUnboundLocalVariable
def train_fn(args):
    # Distribution Strategy
    global model_path, load_model_path, model
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
    optimizer = Adam(learning_rate=args.lr)
    metrics = [dice_weighted, dice_micro]

    # if tf.config.list_physical_devices('GPU'):
    #     strategy = tf.distribute.MirroredStrategy()
    # else:
    #     strategy = tf.distribute.OneDeviceStrategy('/device:CPU:0')

    if args.load_model:
        model_path = paths['save'] + '/' + args.load_model
        model_saves = model_path + '/model_saves'

        assert os.path.exists(model_path)
        load_model_path = sorted(glob(model_saves + '/*'))[-1]
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
        custom_objects = {'custom_loss': custom_loss, 'dice_weighted': dice_weighted, 'dice_micro': dice_micro, 'dice_macro': dice_macro}
        with tf.keras.utils.custom_object_scope(custom_objects):
            print('Model will be loaded from : {}'.format(load_model_path))
            # model = unet(args)
            model = tf.keras.models.load_model(load_model_path)
            # model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)
            # model = tf.keras.models.load_model(load_model_path)                                 # with compile=True ==> RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
        # model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)
            model.compile(optimizer=model.optimizer,
                          loss=custom_loss,
                          metrics=metrics)
        # with compile=False ==> Same
    if not args.load_model or not args.resume:
        model = unet(args)
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)

    learning_rate = lr_schedule(args=args)
    save_model = ModelCheckpoint(filepath=model_saves + '/weights{epoch:03d}', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, save_freq='epoch')
    callbacks = [learning_rate, save_model]

    model.fit(train_dataset,epochs=args.epochs, steps_per_epoch=args.epoch_steps,
              validation_data=eval_dataset, validation_steps=args.eval_steps,
              callbacks=callbacks)