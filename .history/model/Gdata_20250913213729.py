import numpy as np

from csbdeep.models import BaseConfig, BaseModel

from utils.psf import PsfGenerator3D
from utils.zernike import random_zernike_wavefront, ensure_dict

from scipy.signal import convolve
from scipy.ndimage.filters import gaussian_filter
from csbdeep.utils import _raise

import sys
sys.path.append(r'..')

from model.config import Config
from utils.zernike import random_zernike_wavefront
from utils.pseudo import *


def add_random_high_order_disturbances(list_of_coefficients, max_order):

    for i in range(len(list_of_coefficients)):

        random_order = np.random.randint(len(list_of_coefficients[i]), max_order + 1)
        disturbance_size = np.random.normal(0, 0.02) 

        list_of_coefficients[i] = np.append(list_of_coefficients[i], [0] * (random_order - len(list_of_coefficients[i])) if random_order > len(list_of_coefficients[i]) else [])
        list_of_coefficients[i][random_order-1] += disturbance_size

    return list_of_coefficients

class Data:

    def __init__(self,
                 amplitude_ranges, order='noll', normed=True,
                 batch_size=1,
                 psf_shape=(64,64,64), units=(0.1,0.1,0.1), na_detection=1.1, lam_detection=.5, n=1.33, n_threads=4,
                 NoiseIs=None,
                 modulate_aber = None,
                 isMultiStream = None,
                 isRealTime = False,
                 dataFile = None,
                 isRegular = True, regularValue = 1e2
                 ):
        
        self.psfgen = PsfGenerator3D(psf_shape=psf_shape, units=units, lam_detection=lam_detection, 
                                     n=n, na_detection=na_detection, n_threads=n_threads)
        self.order = order
        self.normed = normed
        self.amplitude_ranges = ensure_dict(amplitude_ranges, order)
        self.batch_size = batch_size
        self.NoiseIs = NoiseIs

        self.modulate_aber = modulate_aber
        self.isMultiStream = isMultiStream
        self.isRealTime = isRealTime
        self.dataFile = dataFile
        self.isRegular = isRegular
        self.regularValue = regularValue

    def _single_psf(self):
        
        phi = random_zernike_wavefront(self.amplitude_ranges, order=self.order)        
        return phi.amplitudes_requested,phi.amplitudes_noll
        
    def load_train_data(self):
        print('-'*30)
        print('load train data...')
        print('-'*30)
        print(os.getcwd())
        dataPath = os.path.join(self.dataFile,'data_train.npz')
        if dataPath is not None:
            if os.path.exists(dataPath):
                data = np.load(dataPath)
            else:
                _raise(ValueError("Dataset file not found"))
        else:
            _raise(ValueError("Dataset file path not given"))
        Y_train = data['Y_train']
        Y_valid = data['Y_valid']
        print(Y_train.shape)
        print('load finished')
        num_streams = self.isMultiStream
        X_train = [data[f'X{i}_train'] for i in range(1, num_streams + 1)]
        X_valid = [data[f'X{i}_valid'] for i in range(1, num_streams + 1)]

        return (*X_train, Y_train, *X_valid, Y_valid)
    

    def generator(self,inputs1=None,inputs2=None,inputs3=None,inputs4=None,inputs5=None,inputs6=None,inputs7=None,targets=None,):

        while True:
            ispesudo = 1
            if self.isRealTime:
                amplitudes,amps_noll = zip(*(self._single_psf() for _ in range(self.batch_size)))
                amps_noll = list(amps_noll) 
                # amps_noll = add_random_high_order_disturbances(amps_noll,65)
                # print('High order disturbation')
                PseudoData = PseudoGenerator(psfgen=self.psfgen,amplitudes=amps_noll,
                                             modulate_aber=self.modulate_aber,isMultiStream=self.isMultiStream)
                X = PseudoData.generate_data()  # (num,2,32,32,1)
                Y = np.stack(amplitudes, axis=0) 
                data_dict = {}
                for i in range(min(self.isMultiStream, X.shape[1])):
                    data_dict[f'X{i + 1}'] = X[:, i, :, :, :]
                yield data_dict, Y
             
            else:

                indices = np.arange(len(inputs1))
                np.random.shuffle(indices)
                start_idx = np.random.randint(0,len(inputs1)-self.batch_size)
                excerpt = indices[start_idx:start_idx + self.batch_size]
    
                if self.isMultiStream == 4:
                    yield {'X1':inputs1[excerpt,:,:,:],"X2":inputs2[excerpt,:,:,:],"X3":inputs3[excerpt,:,:,:],"X4":inputs4[excerpt,:,:,:]},targets[excerpt,:]
                elif self.isMultiStream == 3:
                    yield {'X1':inputs1[excerpt,:,:,:],"X2":inputs2[excerpt,:,:,:],"X3":inputs3[excerpt,:,:,:]},targets[excerpt,:]
                elif self.isMultiStream == 2:
                    yield {'X1':inputs1[excerpt,:,:,:],"X2":inputs2[excerpt,:,:,:]},targets[excerpt,:]
                else:
                    yield inputs1[excerpt,:,:,:],targets[excerpt,:]
