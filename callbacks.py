import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

from helper_fns import format_gray_image


class LearningRateLogging(tf.keras.callbacks.Callback):
    def __init__(self, summary_freq='batch', log_freq='epoch'):
        super().__init__()
        self.log_freq = log_freq
        self.summary_freq = summary_freq

    def on_train_batch_begin(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if self.log_freq == 'batch':
            print('\nLearning rate on batch {}: {}'.format(batch, lr))
        if self.summary_freq == 'batch':
            tf.summary.scalar('1-Learning rate / step', lr)
        # print('\nLearning rate for step {} is {}'.format(batch + 1, self.lr))

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        if self.log_freq == 'epoch':
            print('\nLearning rate on epoch {}: {}'.format(epoch, lr))
        if self.summary_freq == 'epoch':
            tf.summary.scalar('1-Learning rate / epoch', lr)


class InputOutputResults(tf.keras.callbacks.Callback):
    def __init__(self, logdir, train_data, eval_data, save_freq='epoch'):
        super().__init__()
        self.data_dict = {'train': train_data, 'eval': eval_data}
        self.logdir = logdir
        self.save_freq = save_freq
        self.val_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        if (isinstance(self.save_freq, int) and
            batch % self.save_freq == 0) or (isinstance(self.save_freq, str) and
                                             self.save_freq == 'batch'):
            self.write_images(mode='train', step=batch)

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self.save_freq, str) and self.save_freq == 'epoch':
            self.write_images(mode='train', step=epoch)
        self.val_epoch = epoch

    def on_test_end(self, logs=None):
        self.write_images(mode='eval', step=self.val_epoch)

    def write_images(self, mode, step):
        assert mode in ['train', 'eval']
        if mode == 'train':
            writer = tf.summary.create_file_writer(logdir=self.logdir + '/{}'.format(mode))
            dataset = self.data_dict['train']
        if mode == 'eval':
            writer = tf.summary.create_file_writer(logdir=self.logdir + '/{}'.format('validation'))
            dataset = self.data_dict['eval']
        image, label = next(dataset.shuffle(200).__iter__())
        output = self.model.predict_on_batch(image)
        image_img = format_gray_image(image)
        output_img = format_gray_image(output)
        label_img = format_gray_image(label)
        with writer.as_default():
            tf.summary.image('1 Input', image_img, step=step, max_outputs=1)
            tf.summary.image('2 Label', label_img, step=step, max_outputs=1)
            tf.summary.image('3 Liver_Pred', tf.expand_dims(output[:, :, :, 1], -1), step=step, max_outputs=1)
            tf.summary.image('4 Output', output_img, step=step, max_outputs=1)

        writer.flush()


class LoadWeightsCallback(tf.keras.callbacks.Callback):
    _chief_worker_only = False

    def __init__(self, weights, optimizer_weights, epoch_steps):
        super().__init__()
        self.weights = weights
        self.optimizer_weights = optimizer_weights
        self.epoch_steps = epoch_steps

    def on_train_begin(self, logs=None):
        self.model.set_weights(self.weights)
        self.model.optimizer.set_weights(self.optimizer_weights)
        initial_epoch = self.model.optimizer.iterations.numpy() // self.epoch_steps
        print('Weights and optimizer state loaded.')
        print('Initial_epoch: {}'.format(initial_epoch))
