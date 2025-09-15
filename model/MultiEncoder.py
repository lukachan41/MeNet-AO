import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from csbdeep.utils import _raise, axes_check_and_normalize, normalize
from csbdeep.models import BaseConfig, BaseModel


def _MultiEncoder(input_shape, output_size, kernel_size, padding, num_inputs=3):
    
    # 定义 resnet block
    def resnet_block(n_filters, kernel_size=kernel_size, batch_norm=True, downsample=False,
                     kernel_initializer="he_normal"):
        def f(inp):
            strides = (2, 2) if downsample else (1, 1)
            x = Conv2D(n_filters, kernel_size, padding='same', use_bias=(not batch_norm),
                       kernel_initializer=kernel_initializer, strides=strides)(inp)
            if batch_norm:
                x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv2D(n_filters, kernel_size, padding=padding, use_bias=(not batch_norm),
                       kernel_initializer=kernel_initializer)(x)
            if batch_norm:
                x = BatchNormalization()(x)

            if strides[0] > 1:
                inp = Conv2D(n_filters, (1, 1), padding=padding, use_bias=(not batch_norm),
                             kernel_initializer=kernel_initializer, strides=strides)(inp)
                if batch_norm:
                    inp = BatchNormalization()(inp)

            x = Add()([inp, x])
            x = PReLU()(x)
            return x

        return f
    
    # 定义 biasEncoder
    def biasEncoder(inputs, stream_index, depth=4):
        n_filters = 16
        x = Conv2D(4, name=f'x{stream_index}conv1', kernel_size=(7, 7), padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(16, name=f'x{stream_index}conv2', kernel_size=(5, 5), padding=padding)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        for i in range(depth):
            x = resnet_block(n_filters * (2 ** i), downsample=(i > 0))(x)
            x = resnet_block(n_filters * (2 ** i))(x)

        x = Conv2D(64, name=f'x{stream_index}conv3', kernel_size=(3, 3), padding=padding)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        return x
    
    # 创建动态数量的输入
    inputs = [Input(input_shape[i], name=f'X{i+1}') for i in range(num_inputs)]

    # 处理每个输入流
    streams = []
    for i, inp in enumerate(inputs):
        streams.append(biasEncoder(inp, stream_index=i+1))
    
    # 对所有输入流的特征进行加权处理并合并
    weighted_streams = [Lambda(lambda x: x)(stream) for stream in streams]

    conct = concatenate(weighted_streams)
    conct = Conv2D(64*num_inputs, name='conct1', kernel_size=kernel_size, padding=padding)(conct)
    conct = BatchNormalization()(conct)
    conct = PReLU()(conct)

    # 继续处理合并后的特征
    n5 = 64*num_inputs
    depth5 = 1
    for i in range(depth5):
        conct = resnet_block(n5 * (2 ** i), downsample=True)(conct)
        conct = resnet_block(n5 * (2 ** i))(conct)

    conct = GlobalAveragePooling2D()(conct)
    d1 = Dense(64*num_inputs, name='dense1')(conct)
    d1 = PReLU()(d1)
    
    output = Dense(output_size, name='Y')(d1)

    # 创建模型
    model = Model(inputs=inputs, outputs=output)
    return model
