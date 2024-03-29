import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, MaxPooling2D, Softmax
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate, Cropping2D


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
    down = LeakyReLU(0.1)(down)
    down = Dropout(rate=dropout)(down)
    down = Conv2D(filters, kernel_size=3, padding=padding)(down)
    if batch_norm:
        down = BatchNormalization()(down)
    down = LeakyReLU(0.1)(down)
    return Dropout(rate=dropout)(down)


def down_block(inputs, filters, dropout, batch_norm, padding='same'):
    connection = double_conv(inputs, filters, dropout, batch_norm, padding=padding)
    down = MaxPooling2D(pool_size=(2, 2), strides=2)(connection)
    return down, connection


def up_block(inputs, connection, filters, dropout, batch_norm, padding='same'):
    up = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding=padding)(inputs)
    if batch_norm:
        up = BatchNormalization()(up)
    up = LeakyReLU(0.1)(up)
    up = Dropout(rate=dropout)(up)
    up = Concatenate()([up, connection])
    return double_conv(up, filters//2, dropout=dropout, batch_norm=batch_norm, padding=padding)


def unet(args, input_shape, strategy):
    with strategy.scope():
        img_inputs = Input(shape=input_shape + [1])
        down, connection_1 = down_block(img_inputs, 32, dropout=args.dropout, batch_norm=args.no_bn)
        down, connection_2 = down_block(down, 64, dropout=args.dropout, batch_norm=args.no_bn)
        down, connection_3 = down_block(down, 128, dropout=args.dropout, batch_norm=args.no_bn)
        down, connection_4 = down_block(down, 256, dropout=args.dropout, batch_norm=args.no_bn)
        bridge = double_conv(inputs=down, filters=1024, dropout=args.dropout, batch_norm=args.no_bn)
        up = up_block(inputs=bridge, connection=connection_4, filters=256, dropout=args.dropout, batch_norm=args.no_bn)
        up = up_block(inputs=up, connection=connection_3, filters=128, dropout=args.dropout, batch_norm=args.no_bn)
        up = up_block(inputs=up, connection=connection_2, filters=64, dropout=args.dropout, batch_norm=args.no_bn)
        up = up_block(inputs=up, connection=connection_1, filters=32, dropout=args.dropout, batch_norm=args.no_bn)
        up = Conv2D(32, kernel_size=3, padding='same')(up)
        up = LeakyReLU()(up)
        up = Conv2D(32, kernel_size=1, padding='same')(up)
        up = LeakyReLU()(up)
        up = Conv2D(2, kernel_size=1, padding='same')(up)
        predict = Softmax(axis=-1)(up)
        model = Model(img_inputs, predict, name='Unet')
        return model


if __name__ == '__main__':
    pass
