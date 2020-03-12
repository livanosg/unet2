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
    def __init__(self, model, logdir, train_data, eval_data, args):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.args = args
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs=None):
        for idx, dataset in enumerate([self.train_data, self.eval_data]):
            if idx == 0:
                mode = 'train'
            else:
                mode = 'eval'
            a = dataset.shuffle(200)
            a = a.take(1).as_numpy_iterator()
            for input, label in a:
                output = self.model.predict(x=input, verbose=1, steps=1)
                writer = tf.summary.create_file_writer(logdir=self.logdir)
                input = (input - np.min(input)) / (np.max(input) - np.min(input)) * 255
                input = input.astype(np.uint8)
                output_img = tf.expand_dims(tf.cast(tf.argmax(output, -1), tf.float32), -1)
                output_pred = output[:, :, :, 1] * 255
                output_pred = tf.cast(output_pred, tf.uint8)
                output_pred = tf.expand_dims(output_pred, -1)
                with writer.as_default():
                    tf.summary.image('{}/Input'.format(mode), input, step=epoch, max_outputs=1)
                    tf.summary.image('{}/Label'.format(mode), tf.expand_dims(label[:, :, :, 1], -1), step=epoch, max_outputs=1)
                    tf.summary.image('{}/Output'.format(mode), output_img, step=epoch, max_outputs=1)
                    tf.summary.image('{}/Liver_Pred'.format(mode), output_pred, step=epoch, max_outputs=1)
                    if self.args.mode == 'test':
                        tf.summary.scalar('Learning_Rate', self.model.optimizer.lr(epoch), step=epoch)
                    writer.flush()
                print('{} images Saved.'.format(mode))
