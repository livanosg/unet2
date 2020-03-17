import os
from os import environ
from glob import glob
from math import ceil
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from archit import unet
from config import paths, root_dir
from input_fns import input_dcm
from loss_fn import custom_loss, weighted_crossentropy, weighted_log_dice_loss
from lr_schedules import lr_schedule
from metrics import dice_weighted, dice_micro, dice_macro
from callbacks import LearningRateLogging, ShowImages, MetricsSummaries
from paths import get_paths


def train_fn(args):
    # Distribution Strategy
    environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + root_dir
    environ['XLA_FLAGS'] = "--xla_dump_to=/home/medphys/projects/unet2/dumbs"
    device = 'CPU'
    if args.distributed and tf.config.list_physical_devices('GPU'):
        if environ['CUDA_VISIBLE_DEVICES'] is None:
            environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = 'GPU'
        tf.config.set_soft_device_placement(enabled=True)
        print('Mirrored distribution strategy selected.')
        num_devices = len(tf.config.list_physical_devices(device))
        strategy = tf.distribute.MirroredStrategy()
    else:
        print('One device strategy selected.')
        num_devices = len(tf.config.list_physical_devices(device))
        strategy = tf.distribute.OneDeviceStrategy(device)
    global_batch_size = args.batch_size * num_devices
    print('Number of {} devices available: {}'.format(device, num_devices))

    if args.modality in ('CT', 'ALL'):
        input_shape = [512, 512]
    else:
        input_shape = [320, 320]

    if args.mode == 'testing':
        train_size = 10
        eval_size = 5
        args.epoch_steps = ceil(train_size / global_batch_size)
        args.eval_steps = eval_size
    else:
        args.epoch_steps = ceil(len(get_paths('train', args.modality))/global_batch_size)
        args.eval_steps = len(get_paths('eval', args.modality))

    train_dataset = input_dcm(mode='train', args=args)
    eval_dataset = input_dcm(mode='eval', args=args)

    optimizer = Adam(learning_rate=args.lr)
    metrics = [dice_micro]

    init_epoch = 0
    if args.load_model:
        model_path = paths['save'] + '/' + args.load_model
        model_saves = model_path + '/model_saves'
        assert os.path.exists(model_path)
        load_model_path = sorted(glob(model_saves + '/*'))[-1]
        print('Model will be loaded from : {}'.format(load_model_path))
        init_epoch = int(load_model_path[-3:])

        custom_objects = {'custom_loss': custom_loss, 'weighted_crossentropy': weighted_crossentropy,
                          'weighted_log_dice_loss': weighted_log_dice_loss, 'dice_weighted': dice_weighted,
                          'dice_micro': dice_micro, 'dice_macro': dice_macro}
        with tf.keras.utils.custom_object_scope(custom_objects):
            with strategy.scope():
                print('Loading model...')
                model = tf.keras.models.load_model(load_model_path)
                print('Model loaded!')
        if not args.resume:
            with strategy.scope():
                config = model.get_config()
                weights = model.get_weights()
                model = tf.keras.Model.from_config(config)
                model.set_weights(weights)
    else:
        with strategy.scope():
            model = unet(args, input_shape, strategy)

    if (not args.load_model) or (not args.resume):
        with strategy.scope():
            model.compile(optimizer=optimizer, loss=weighted_crossentropy, metrics=metrics)
        print('New training session')
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
        model_saves = model_path + '/model_saves'
        os.makedirs(model_saves, exist_ok=True)

    lr_log = LearningRateLogging(model_path=model_path)
    learning_rate = lr_schedule(args=args)

    show_images = ShowImages(model_path=model_path, args=args, input_shape=input_shape)
    show_summaries = MetricsSummaries(model_path=model_path, args=args)
    early_stopping = EarlyStopping(patience=args.early_stop, verbose=1, mode='min')
    save_model = ModelCheckpoint(filepath=model_saves + '/weights{epoch:03d}', verbose=1, save_best_only=True)
    callbacks = [lr_log, learning_rate, save_model, early_stopping, show_images, show_summaries]

    model.fit(train_dataset, epochs=args.epochs, initial_epoch=init_epoch, steps_per_epoch=args.epoch_steps,
              validation_data=eval_dataset, validation_steps=args.eval_steps,
              callbacks=callbacks)
