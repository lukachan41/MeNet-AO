import numpy as np
import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from csbdeep.models import BaseModel


import sys
sys.path.append(r'..')

from model.config import Config
from utils.utils import *
from model.MultiEncoder import _MultiEncoder
from utils.pseudo import *
from .Gdata import Data


class MeNet(BaseModel):
    """
        MeNet model.

    """

    @property
    def _config_class(self):
        return Config
    @property
    def _axes_out(self):
        return 'C'
    def get_model_input_shape(self):
        base_shape = tuple((32, 32, 2))
        if self.config.isMultiStream >= 2:
            self.config.input_shape  = [base_shape] * self.config.isMultiStream
        elif self.config.isMultiStream == 1:
            self.config.input_shape  =  tuple((32, 32, 2*len(self.config.modulate_aber)))
        return self.config.input_shape 

    def _build(self):
    
        input_shape =  self.get_model_input_shape()

        output_size = self.config.n_channel_out
        kernel_size = self.config.net_kernel_size
        pool_size = self.config.net_pool_size
        activation = self.config.net_activation
        padding = self.config.net_padding

        if self.config.isMultiStream == 0:
            encoderNum = 1
        else:
            encoderNum = self.config.isMultiStream
        print('input_shape',input_shape)
        print('output_size:',+output_size)
        if self.config.net_architecture == 'singleEncoder':
            return self._singleEncoder(input_shape, output_size, kernel_size, padding)
        elif self.config.net_architecture == 'MeNet':
            return _MultiEncoder(input_shape, output_size, kernel_size, padding,  encoderNum)

    def _singleEncoder(self, input_shape, output_size, kernel_size, padding):

        def resnet_block(n_filters, kernel_size=kernel_size, batch_norm=True, downsample=False,
                        kernel_initializer="he_normal"):
            def f(inp):
                strides = (2, 2) if downsample else (1, 1)
                x = Conv2D(n_filters, kernel_size, padding='same', use_bias=(not batch_norm),
                           kernel_initializer=kernel_initializer, strides=strides,
                           )(inp)
                if batch_norm:
                    x = BatchNormalization()(x)
                #x = Activation(activation)(x)
                x = PReLU()(x)

                x = Conv2D(n_filters, kernel_size, padding=padding, use_bias=(not batch_norm),
                           kernel_initializer=kernel_initializer,
                           )(x)
                if batch_norm:
                    x = BatchNormalization()(x)

                if downsample:
                    inp = Conv2D(n_filters, (1, 1), padding=padding, use_bias=(not batch_norm), 
                                kernel_initializer=kernel_initializer,strides=strides,
                                )(inp)
                    if batch_norm:
                        inp = BatchNormalization()(inp)

                x = Add()([inp, x])
                #x = Activation(activation)(x)
                x = PReLU()(x)
                return x

            return f

        inp = Input(input_shape, name='X')
        x = inp
        x = Conv2D(2,name='x1conv1', kernel_size=(7,7),padding=padding,)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Conv2D(16,name='x1conv2', kernel_size=(5,5),padding=padding,)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
 
        n = 16
            
        depth = 5
        extra_blocks = [0, 0, 0, 0, 1,1]
        for i in range(depth):
            x = resnet_block(n * (2 ** i), downsample=(i > 0))(x)
            x = resnet_block(n * (2 ** i))(x)
            # for _ in range(extra_blocks[i]):            # 追加的普通 block
            #     x = resnet_block(n * (2 ** i))(x)
        x = resnet_block(256)(x)
        x = Conv2D(192,name='x1conv3', kernel_size=(3,3),padding=padding,)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        
        x = GlobalAveragePooling2D()(x)
        # x = Dense(32, name='dense1')(x)
        x = Dense(192, name='dense1')(x)
        x = PReLU()(x)
        oup = Dense(output_size, name='Y')(x)

        return Model(inp, oup)
    def prepare_data(self,data_val, data_train, n_streams):

        train_data = data_val.load_train_data()
        half = len(train_data) // 2
        train_split, valid_split = train_data[:half], train_data[half:]

        Y_train, Y_valid = train_split[-1], valid_split[-1]
        X_train, X_valid = train_split[:-1], valid_split[:-1]


        inputs_train = {f"inputs{i+1}": X_train[i] for i in range(n_streams)}
        inputs_valid = {f"inputs{i+1}": X_valid[i] for i in range(n_streams)}

        inputs_train["targets"] = Y_train
        inputs_valid["targets"] = Y_valid

        data_val_gen = data_val.generator(**inputs_valid)
        data_train_gen = data_train.generator(**inputs_train)
        return data_val_gen, data_train_gen
    
    def prepare_for_training(self, optimizer=None):

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(lr=self.config.train_learning_rate )
        self.keras_model.compile(optimizer, loss=self.config.train_loss)
        self.callbacks = []
        if self.basedir is not None:
            print('trainCheckpoint',self.config.train_checkpoint)
            self.callbacks += self._checkpoint_callbacks()
            if self.config.train_tensorboard:
                self.callbacks.append(TensorBoard(log_dir=str(self.logdir), write_graph=False))
        self._model_prepared = True


    def train(self, seed=None, epochs=None, steps_per_epoch=None):
        if seed is not None:
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch
        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            amplitude_ranges     = self.config.zernike_amplitude_ranges,
            order                = self.config.zernike_order,
            normed               = self.config.zernike_normed,
            psf_shape            = self.config.psf_shape,
            units                = self.config.psf_units,
            na_detection         = self.config.psf_na_detection,
            lam_detection        = self.config.psf_lam_detection,
            n                    = self.config.psf_n,
            NoiseIs              = self.config.NoiseIs,
            modulate_aber        = self.config.modulate_aber,
            isMultiStream        = self.config.isMultiStream,
            isRealTime           = self.config.isRealTime,
            dataFile             = self.config.dataFile,
            isRegular            = self.config.isRegular,
            regularValue         = self.config.regularValue
        )

        data_train = Data(batch_size=self.config.train_batch_size, **data_kwargs)
        data_val = Data(batch_size=self.config.train_n_val, **data_kwargs) 
        
        if self.config.isRealTime is False:
            data_val, G = self.prepare_data(data_val, data_train, self.config.isMultiStream)
        else:
            # real-time data generation is too slow on the fly
            data_val = next(data_val.generator())       
            G = data_train.generator()
            
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        history = self.keras_model.fit_generator(generator=G, validation_data=data_val,
                                                epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                validation_steps=1,
                                                callbacks=[self.callbacks,
                                                    reduce_lr], 
                                                verbose=1)

        self._training_finished()
        return history
