import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Softmax
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate, Cropping2D
from tensorflow.keras.utils import plot_model


def crop_con(up_layer, down_layer):
    up_shape = tf.shape(up_layer)
    down_shape = tf.shape(down_layer)
    offsets = [0, (down_shape[1] - up_shape[1]) // 2, (down_shape[2] - up_shape[2]) // 2, 0]
    size = [-1, up_shape[1], up_shape[2], 1]
    Cropping2D()
    down_cropped = tf.slice(down_layer, offsets, size)
    return tf.concat([down_cropped, up_layer], -1)  # Concatenate at number of feature maps axis.


def double_conv(inputs, filters, dropout, batch_norm, padding='same'):
    down = Conv2D(filters, kernel_size=3, padding=padding)(inputs)
    if batch_norm:
        down = BatchNormalization()(down)
    down = tf.keras.layers.LeakyReLU()(down)
    down = Dropout(rate=dropout)(down)
    down = Conv2D(filters, kernel_size=3, padding=padding)(down)
    if batch_norm:
        down = BatchNormalization()(down)
    down = tf.keras.layers.LeakyReLU()(down)
    return Dropout(rate=dropout)(down)


def down_block(inputs, filters, dropout, batch_norm, padding='same'):
    connection = double_conv(inputs, filters, dropout, batch_norm, padding=padding)
    down = MaxPooling2D()(connection)
    return down, connection


def up_block(inputs, connection, filters, dropout, batch_norm, padding='same'):
    up = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='valid')(inputs)
    if batch_norm:
        up = BatchNormalization()(up)
    up = tf.keras.layers.LeakyReLU()(up)
    up = Dropout(rate=dropout)(up)
    up = Concatenate()([up, connection])
    up = double_conv(up, filters//2, dropout, batch_norm, padding=padding)
    return double_conv(up, filters//2, dropout, batch_norm, padding=padding)


def unet(dropout, batch_norm):
    padding = 'same'
    img_inputs = Input([None, None, 1])
    with tf.name_scope('Down_1'):
        out, connection_1 = down_block(img_inputs, 32, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Down_2'):
        out, connection_2 = down_block(out, 64, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Down_3'):
        out, connection_3 = down_block(out, 128, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Down_4'):
        out, connection_4 = down_block(out, 256, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Bridge'):
        bridge = double_conv(inputs=out, filters=1024, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Up_1'):
        up = up_block(inputs=bridge, connection=connection_4, filters=256, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Up_2'):
        up = up_block(inputs=up, connection=connection_3, filters=128, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Up_3'):
        up = up_block(inputs=up, connection=connection_2, filters=64, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Up_4'):
        up = up_block(inputs=up, connection=connection_1, filters=32, dropout=dropout, batch_norm=batch_norm, padding=padding)
    with tf.name_scope('Output'):
        up = Conv2D(32, kernel_size=3, padding=padding)(up)
        up = tf.keras.layers.LeakyReLU()(up)
        up = Conv2D(32, kernel_size=1, padding=padding)(up)
        up = tf.keras.layers.LeakyReLU()(up)
        out = Conv2D(2, kernel_size=1)(up)
        predict = Softmax(axis=-1)(out)
    return Model(img_inputs, predict, name='Unet')


if __name__ == '__main__':
    model = unet(dropout=0.5, batch_norm=False)
    model.summary()
    plot_model(model, 'model.png', show_shapes=True)