import sys
sys.path.append(r'..') 
import os
from model.Gdata import Data
from model.config import Config
import numpy as np
import time

import json

def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        line = json.dumps(data,**kwargs)
        f.write(line+'\n')

zern = [5,6,7,8,9,10,11]   # Initial aberration modes
amp_range = [0.5]*len(zern)   # Initial aberration amplitudes
amps = dict(zip(zern, amp_range))  
modulationAber = {5:0.2,6:0.2,11:0.2} # Modulated aberrations

# zern = [5]   # Initial aberration modes
# amp_range = [0.5]*len(zern)   # Initial aberration amplitudes
# amps = dict(zip(zern, amp_range))  
# modulationAber = {5:0.2} # Modulated aberrations

isRegular = True
regularValue = 1e2
dataFile = '../DataSet/Demo3//' 
c = Config(zernike_amplitude_ranges=amps,modulate_aber=modulationAber,
           isMultiStream = len(modulationAber),isRealTime = True,
           psf_na_detection=1.05, psf_units=(0.1,0.086,0.086), psf_n=1.33, psf_lam_detection=0.920,
           dataFile = dataFile, isRegular = isRegular, regularValue = regularValue,NoiseIs=False)
#vars(c)

data_kwargs = dict (
            amplitude_ranges     = c.zernike_amplitude_ranges,
            order                = c.zernike_order,
            normed               = c.zernike_normed,
            psf_shape            = c.psf_shape,
            units                = c.psf_units,
            na_detection         = c.psf_na_detection,
            lam_detection        = c.psf_lam_detection,
            n                    = c.psf_n,
            NoiseIs              = c.NoiseIs,
            noise_Q              = c.noise_Q,
            noise_sigma          = c.noise_sigma,
            modulate_aber        = c.modulate_aber,
            isMultiStream        = c.isMultiStream,
            isRealTime           = c.isRealTime,
            dataFile            = c.dataFile,
            isRegular           = c.isRegular,
            regularValue        = c.regularValue
        )

data_val = Data(batch_size=10000, **data_kwargs)

print('Initial aberrations',zern)
print('Modulation Aberrations',modulationAber)
print('Start')
start = time.perf_counter() 

data_val = next(data_val.generator())

end1 = time.perf_counter()
print("Generated Time : %s Seconds"%(end1-start))

X = data_val[0]
Y = data_val[1]

print(type(X))
print(type(Y))
 
X1 = X['X1']
X2 = X['X2']
X3 = X['X3']
# print(X1.shape)
# print(X2.shape)
# print(X3.shape)
if not os.path.exists(dataFile):
    os.makedirs(dataFile)

np.save(dataFile+'/X1.npy',X1)
np.save(dataFile+'X2.npy',X2)
np.save(dataFile+'X3.npy',X3)
np.save(dataFile+'/Y.npy',Y)


L = len(X1)
indices = np.random.permutation(L)
train_end = int(L * 0.8)
valid_end = int(L * 0.9)
print("shuffle")

np.savez(os.path.join(dataFile, "data_train.npz"),   
    X1_train=X1[indices[indices[:train_end]]],X2_train=X2[indices[indices[:train_end]]],
    X3_train=X3[indices[indices[:train_end]]],Y_train=Y[indices[indices[:train_end]]],
    X1_valid=X1[indices[train_end:valid_end]],X2_valid=X2[indices[train_end:valid_end]],
    X3_valid=X3[indices[train_end:valid_end]],Y_valid=Y[indices[train_end:valid_end]])

np.savez(os.path.join(dataFile, "data_test.npz"),   
    X1_test=X1[indices[valid_end:]],X2_test = X2[indices[valid_end:]],
    X3_test=X3[indices[valid_end:]],Y_test=Y[indices[valid_end:]])
# X = np.concatenate((X1,X2),axis=-1)

# np.savez(os.path.join(dataFile, "data_train.npz"),   
    # X1_train=X1[indices[:train_end]],Y_train=Y[indices[:train_end]],
    # X1_valid=X1[indices[train_end:valid_end]],Y_valid=Y[indices[train_end:valid_end]])
# np.savez(os.path.join(dataFile, "data_test.npz"),   
#     X1_test=X1[indices[valid_end:]],Y_test=Y[indices[valid_end:]])

save_json(vars(c), str(dataFile+'/config.json'))
end2 = time.perf_counter()

print("Running time:: %s Seconds"%(end2-start))
