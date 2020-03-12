import os
import tensorflow as tf
import tensorflow_addons as tfa
from os import environ
from math import ceil
from tensorflow_addons import metrics
from tensorflow.keras.optimizers import Adam
from archit import unet
from callbacks import PrintLR, InputOutputResults
from data_generators import data_gen
from input_fns import train_eval_input_fn
from config import paths
from loss_fn import custom_loss
from metrics import DiceScore


def train_fn(args):
    # Distribution Strategy
    environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # TODO Implement on multi-nodes SLURM
    strategy = tf.distribute.MirroredStrategy()
    tf.config.set_soft_device_placement(enabled=True)  # Choose CPU if no GPu
    # session_config = tf.config.allow_growth = True  # Allow full memory usage of GPU.\
    if args.load_model:
        model_path = paths['save'] + '/' + args.load_model
        logs_path = model_path + '/logs/'
    else:
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
        logs_path = model_path + '/logs/'
    print('Model saved at: {}'.format(model_path))
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
                                                                  step_size=2 * steps_per_epoch)
    optimizer = Adam(learning_rate=learning_rate)

    # Define metrics
    f1_score = metrics.F1Score(args.classes, 'micro')
    dice_score = DiceScore(num_classes=2, average='micro')


    with strategy.scope():
        model = unet(dropout=0.5, batch_norm=True)
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=[dice_score, f1_score])
    tf.summary.image('Input', model.input[0], step=model.optimizer.iterations)
    tf.summary.image('Output', tf.expand_dims(tf.cast(tf.argmax(model.outputs[0]), tf.float32), axis=-1),
                     step=model.optimizer.iterations)
    # Define  callbacks.

    lr_cb = PrintLR(model=model)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path + '/', monitor='val_loss',
                                                    save_weights_only=True, save_best_only=True, mode='min', verbose=1)
    gianadoumegamwthnpanagia = InputOutputResults(model=model, logdir=logs_path, dataset=train_dataset, args=args)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(x=train_dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
              validation_data=eval_dataset, validation_steps=eval_size,
              callbacks=[tensorboard_callback, checkpoint, lr_cb, gianadoumegamwthnpanagia, early_stop])
