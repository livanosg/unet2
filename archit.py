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


def unet(args):
    if args.modality in ('CT', 'ALL'):
        input_shape = [512, 512, 1]
    else:
        input_shape = [320, 320, 1]
    padding = 'same'
    img_inputs = Input(shape=input_shape)
    block_1 = Model(inputs=img_inputs,outputs=[down_block(img_inputs, 32, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)], name="Block_1")
    block_2 = Model(inputs=block_1.output[0],outputs=[down_block(block_1.output[0], 64, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)], name="Block_2")
    block_3 = Model(inputs=block_2.output[0],outputs=[down_block(block_2.output[0], 128, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)], name="Block_3")
    block_4 = Model(inputs=block_3.output[0],outputs=[down_block(block_3.output[0], 256, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)], name="Block_4")
    bridge = Model(inputs=block_4.output[0],outputs=[double_conv(inputs=block_4.output[0], filters=1024, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)], name="Block_4")
    # bridge = double_conv(inputs=out, filters=1024, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)
    # up = up_block(inputs=bridge, connection=connection_4, filters=256, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)
    # up = up_block(inputs=up, connection=connection_3, filters=128, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)
    # up = up_block(inputs=up, connection=connection_2, filters=64, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)
    # up = up_block(inputs=up, connection=connection_1, filters=32, dropout=args.dropout, batch_norm=args.no_bn, padding=padding)
    # up = Conv2D(32, kernel_size=3, padding=padding)(up)
    # up = tf.keras.layers.LeakyReLU()(up)
    # up = Conv2D(32, kernel_size=1, padding=padding)(up)
    # up = tf.keras.layers.LeakyReLU()(up)
    # out = Conv2D(2, kernel_size=1)(up)
    # predict = Softmax(axis=-1)(out)
    return Model(img_inputs, block_1.outputs, name='Unet')


if __name__ == '__main__':
    model = unet(dropout=0.5, batch_norm=False)
    model.summary()
    plot_model(model, 'model.png', show_shapes=True)