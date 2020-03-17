import tensorflow as tf
from helper_fns import format_gray_image
from input_fns import input_dcm


class LearningRateLogging(tf.keras.callbacks.Callback):
    def __init__(self, model_path, summary_freq='epoch', log_freq='epoch'):
        super().__init__()
        self.log_freq = log_freq
        self.summary_freq = summary_freq
        self.train_writer = tf.summary.create_file_writer(logdir=model_path + '/train')

    def on_train_batch_begin(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if self.log_freq == 'batch':
            print('\nLearning rate on batch {}: {}'.format(batch, lr))
        if self.summary_freq == 'batch':
            with self.train_writer.as_default():
                tf.summary.scalar('Learning rate', lr, step=batch)
            self.train_writer.flush()
        # print('\nLearning rate for step {} is {}'.format(batch + 1, self.lr))

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        if self.log_freq == 'epoch':
            print('\nLearning rate on epoch {}: {}'.format(epoch + 1, lr))
        if self.summary_freq == 'epoch':
            with self.train_writer.as_default():
                tf.summary.scalar('Learning rate', lr, step=epoch)
            self.train_writer.flush()


# noinspection PyUnboundLocalVariable
class ShowImages(tf.keras.callbacks.Callback):
    def __init__(self, model_path, args, input_shape, save_freq='epoch'):
        super().__init__()
        self.dataset_fn = input_dcm
        self.args = args
        self.save_freq = save_freq
        self.val_epoch = 0
        self.train_writer = tf.summary.create_file_writer(logdir=model_path + '/train')
        self.val_writer = tf.summary.create_file_writer(logdir=model_path + '/validation')
        self.input_shape = input_shape

    def on_train_batch_end(self, batch, logs=None):
        if (isinstance(self.save_freq, int) and
            batch % self.save_freq == 0) or (isinstance(self.save_freq, str) and
                                             self.save_freq == 'batch'):
            self.write_images(mode='train', step=batch, writer=self.train_writer)

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self.save_freq, str) and self.save_freq == 'epoch':
            self.write_images(mode='train', step=epoch, writer=self.train_writer)
            self.write_images(mode='eval', step=epoch, writer=self.val_writer)
        self.val_epoch = epoch

    def on_test_end(self, logs=None):
        self.write_images(mode='eval', step=self.val_epoch, writer=self.val_writer)

    def write_images(self, mode, step, writer):
        for element in self.dataset_fn(mode=mode, args=self.args).take(1):
            image, label = element[0], element[1]
        output = self.model.predict_on_batch(image)
        images = {'1 Input': format_gray_image(image),
                  '2 Label': format_gray_image(label),
                  '3 Liver_Pred': tf.expand_dims(output[:, :, :, 1], -1),
                  '4 Output': format_gray_image(output)}
        with writer.as_default():
            if mode == 'eval':
                mode = 'validation'
            for keys, values in images.items():
                tf.summary.image(mode + '/' + keys, values, step=step, max_outputs=1)
        writer.flush()


# noinspection PyMethodOverriding
class MetricsSummaries(tf.keras.callbacks.Callback):
    def __init__(self, model_path, args, summary_freq='epoch'):
        super().__init__()
        self.summary_freq = summary_freq
        self.train_writer = tf.summary.create_file_writer(logdir=model_path + '/train')
        self.val_writer = tf.summary.create_file_writer(logdir=model_path + '/validation')
        self.epoch = 0
        self.batch = 0
        self.epoch_steps = args.epoch_steps

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = self.epoch * self.epoch_steps + batch

    def on_epoch_end(self, epoch, logs):
        if self.summary_freq == 'epoch':
            for key, value in logs.items():
                if 'val_' in key:
                    with self.val_writer.as_default():
                        if key not in ('loss', 'val_loss'):
                            tf.summary.scalar(name='Metrics/' + key.replace('val_', ''), data=value, step=self.epoch)
                        else:
                            tf.summary.scalar(name=key.replace('val_', ''), data=value, step=self.epoch)
                else:
                    with self.train_writer.as_default():
                        if key not in ('loss', 'val_loss'):
                            tf.summary.scalar(name='Metrics/' + key, data=value, step=self.epoch)
                        else:
                            tf.summary.scalar(name=key, data=value, step=self.epoch)

            self.val_writer.flush()
            self.train_writer.flush()
