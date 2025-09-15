from csbdeep.models import BaseConfig, BaseModel
from phasenet.zernike import random_zernike_wavefront, ensure_dict

class Config(BaseConfig):

    def __init__(self, axes='ZYX', n_channel_in=1, **kwargs):


        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1)

        self.zernike_amplitude_ranges  = {'vertical coma': (-0.2,0.2)}
        self.zernike_order             = 'noll'
        self.zernike_normed            = True

        self.net_architecture         = "convnet"
        self.net_kernel_size           = (3,3)
        self.net_pool_size             = (2,2)
        self.net_activation            = 'tanh'
        self.net_padding               = 'same'

        self.psf_shape                 = (64,64,64)
        self.psf_units                 = (0.1,0.1,0.1)
        self.psf_na_detection          = 1.1
        self.psf_lam_detection         = 0.5
        self.psf_n                     = 1.33
        self.NoiseIs                   = False


        self.train_loss                = 'mse'
        self.train_epochs              = 200
        self.train_steps_per_epoch     = 1000
        self.train_learning_rate       = 6e-4
        self.train_batch_size          = 32
        self.train_n_val               = 16   
        self.train_tensorboard         = True


        # Modulation aberrations
        self.modulate_aber              = None
        # multi-stream or three-D
        self.isMultiStream              = 1    
        # real-time data genertaion
        self.isRealTime                 = False
        self.isRegular                  = True
        self.regularValue               = 1e2
        self.dataFile                   = None
        self.input_shape                 = None

        
        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        self.n_channel_out = len(random_zernike_wavefront(self.zernike_amplitude_ranges))
        if self.isMultiStream:
            self.input_shape = [tuple((32, 32, 2))]*self.isMultiStream
        else:
            self.input_shape = tuple((32, 32, 2*len(self.bias_aber)))