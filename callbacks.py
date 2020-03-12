import numpy as np
import tensorflow as tf


class PrintLR(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                          self.model.optimizer.lr(epoch)))


class InputOutputResults(tf.keras.callbacks.Callback):
    def __init__(self, model, logdir, dataset, args):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.args = args
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs=None):
        a = self.dataset.shuffle(200)
        a = a.take(1).as_numpy_iterator()
        for input, label in a:
            output = self.model.predict(x=input, verbose=1, steps=1)
            writer = tf.summary.create_file_writer(logdir=self.logdir)
            input = (input - np.min(input)) / (np.max(input) - np.min(input)) * 255
            input = input.astype(np.uint8)
            output = output[:, :, :, 1] * 255
            output = tf.cast(output, tf.uint8)
            output = tf.expand_dims(output, -1)
            with writer.as_default():
                tf.summary.image('{}/Input'.format(self.args.mode), input, step=epoch, max_outputs=1)
                tf.summary.image('{}/Label'.format(self.args.mode), tf.expand_dims(label[:, :, :, 1], -1), step=epoch, max_outputs=1)
                tf.summary.image('{}/Output'.format(self.args.mode), tf.expand_dims(tf.cast(tf.argmax(output, -1), tf.float32), -1), step=epoch, max_outputs=1)
                tf.summary.image('{}/Liver_Pred'.format(self.args.mode), output, step=epoch, max_outputs=1)
                if self.args.mode == 'train':
                    tf.summary.scalar('Learning_Rate', self.model.optimizer.lr(epoch))

                writer.flush()
        writer.close()
        print('\nImages Saved.')
