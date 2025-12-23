import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(r'..')
import cv2
from scipy.signal import convolve
from utils.zernike import ZernikeWavefront
from utils.noise import add_poisson_gaussian_noise_np


def rgb2gray(rgb):

  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def display_frequency_spectrum(img):

    xf = np.fft.fft2(img)
    xfshift = np.fft.fftshift(xf)
    fimg = np.log1p(np.abs(xfshift))  
    fimg_normalized = np.log1p(np.abs(xfshift))
    fimg_normalized = (fimg_normalized - np.min(fimg_normalized)) / (np.max(fimg_normalized) - np.min(fimg_normalized))
    
    return fimg_normalized;
    
    return fimg_normalized
def spectrum_show(img):
    xf = np.fft.fft2(img)
    xfshift = np.fft.fftshift(xf)
    fimg = np.log(np.abs(xfshift))
    plt.figure()
    plt.imshow(fimg, cmap='Greys_r');
    return fimg

def intensity_img(spectrum_img):
    inverse_shift = np.fft.ifftshift(spectrum_img)
    iimg = np.fft.ifft2(inverse_shift)
    return iimg

from scipy.ndimage import gaussian_filter
def high_pass_filter(img, sigma=5):
    return img - gaussian_filter(img, sigma=sigma)

def Pseudo_fft(img1,img2):
    fft_img1 = np.fft.fftshift(np.fft.fft2(img1))
    fft_img2 = np.fft.fftshift(np.fft.fft2(img2))
    pesudo = np.fft.ifftshift(np.fft.ifft2(fft_img1/(fft_img2+1e-6)))
    return np.abs(pesudo)

def pseudo_ffttest_tikhonov(img1, img2, alpha=0.01):
    img1 = high_pass_filter(img1)
    img2 = high_pass_filter(img2)
    
    fft_img1 = np.fft.fftshift(np.fft.fft2(img1))
    fft_img2 = np.fft.fftshift(np.fft.fft2(img2))
    fft_ratio = (fft_img1 * np.conj(fft_img2)) / (np.abs(fft_img2)**2 + alpha)
    pseudo_img = np.fft.ifftshift(np.fft.ifft2(fft_ratio))
    
    return np.abs(pseudo_img)


def extract_center(img,edge=16):
    l = img.shape[0]
    w = img.shape[1]
    crop_psf = img[int(l/2-edge):int(l/2+edge),int(w/2-edge):int(w/2+edge)]
    return crop_psf
def extract_center_withShift(img, edge=16, *, use_abs=True, pad_mode='reflect', return_center=False):

    arr = np.asarray(img)
    if arr.ndim < 2:
        raise ValueError("img must be 2D (H, W[,...])")
    if arr.ndim == 2:
        base = np.abs(arr) if use_abs else arr
    else:
        chwise = np.abs(arr) if use_abs else arr
        base = chwise.mean(axis=-1) 

    signal = base.astype(float)
    signal[np.isnan(signal)] = -np.inf
    r, c = np.unravel_index(signal.argmax(), signal.shape)

    pad_width = [(edge, edge), (edge, edge)] + [(0, 0)] * (arr.ndim - 2)
    arr_p = np.pad(arr, pad_width, mode=pad_mode)
    rp, cp = r + edge, c + edge
    crop = arr_p[rp - edge: rp + edge, cp - edge: cp + edge, ...]

    if return_center:
        return crop, (r, c)
    return crop

class PseudoGenerator:
    """
    Pseudo-psf generator
    """
    def __init__(self, psfgen=None,amplitudes=None, modulate_aber=None,isMultiStream = None,
                 img_file_path='../Data/img/',isRegular = True, regularValue = 1e2,
                 NoiseIs = False, noise_Q = 10000,noise_sigma = 1000):

        self.psfgen = psfgen
        self.amplitudes = amplitudes
        self.modulate_aber = modulate_aber
        self.img_file_path = img_file_path
        self.isMultiStream = isMultiStream
        self.isRegular = isRegular
        self.regularValue = regularValue
        self.NoiseIs = NoiseIs
        self.noise_Q = noise_Q
        self.noise_sigma = noise_sigma

    def obtain_img(self):
        img_path = self.img_file_path
        file_paths = [os.path.join(img_path,f) for f in os.listdir(img_path)]
        random_file_path = random.choice(file_paths)
        #print(random_file_path)
        img = cv2.imread(random_file_path)[:,:,(2,1,0)]
        img = img[:,:,0]
        img = img.astype(float)
        min_val = np.min(img)
        max_val = np.max(img)

        normalized_image = (img - min_val) / (max_val - min_val)
        return normalized_image
    
    def generate_data(self):
        outlist = []

        max_aber = 0
        for k in self.modulate_aber.keys():
            if isinstance(k,str):
                k = int(k)
            max_aber = max(max_aber,k)
        # print(len(self.amplitudes))
        # print('bias',len( self.bias_aber.items()))
        for amps in self.amplitudes:
            # amps:tuple
            # amps_tmp: list
            amps_tmp = list(amps)
            if len(amps_tmp) < max_aber:
                amps_tmp = amps_tmp + (max_aber-len(amps_tmp)+1)*[0]
            single_out = []
            for k,v in self.modulate_aber.items():
    
                if isinstance(k,str):
                    k = int(k)
                positive_amps = amps_tmp
                positive_amps[k-1] += v  # + bias aberration
                positive_amps = dict(zip(list(range(1,len(positive_amps)+1)), positive_amps))
                negative_amps = amps_tmp
                negative_amps[k-1] -= v  # - bias aberration
                negative_amps = dict(zip(list(range(1,len(negative_amps)+1)), negative_amps))

                wf_positive = ZernikeWavefront(positive_amps, order='noll')
                h_positive = self.psfgen.incoherent_psf(wf_positive, normed=False)

                wf_negative = ZernikeWavefront(negative_amps, order='noll')
                h_negative = self.psfgen.incoherent_psf(wf_negative, normed=False)

                mid_plane_negative = h_negative.shape[0]//2;

                img = self.obtain_img()

                img_positive = convolve(img,h_positive[mid_plane_negative],'same')
                img_negative = convolve(img,h_negative[mid_plane_negative],'same')

                img_positive = (img_positive-np.min(img_positive))/(np.max(img_positive)-np.min(img_positive))
                img_negative = (img_negative-np.min(img_negative))/(np.max(img_negative)-np.min(img_negative))
                
                # Noise modeling
                rng = np.random.default_rng(0)
                def mpg_sample(mean_dn, Q, sigma, n=16):
                    max_dn = (2**n - 1)
                    mean_dn = mean_dn * max_dn
                    lam = (Q / max_dn) * np.clip(mean_dn, 0, max_dn)
                    k = rng.poisson(lam=lam)
                    dn = (max_dn / Q) * k + rng.normal(0, sigma, mean_dn.shape)
                    return np.clip(dn, 0, max_dn)
                if self.NoiseIs:
                    Q0 = self.noise_Q
                    sigma0 = self.noise_sigma
                    img_positive = mpg_sample(img_positive,Q=Q0,sigma=sigma0)
                    img_negative = mpg_sample(img_negative,Q=Q0,sigma=sigma0)
                
                
                img_positive = (img_positive-np.min(img_positive))/(np.max(img_positive)-np.min(img_positive))
                img_negative = (img_negative-np.min(img_negative))/(np.max(img_negative)-np.min(img_negative))
                
                if self.isRegular:
                    Pseudo_12 = extract_center_withShift(Pseudo_fft(img_positive,img_negative))
                    Pseudo_21 = extract_center_withShift(Pseudo_fft(img_negative,img_positive))
                else:
                    Pseudo_12 = extract_center_withShift(pseudo_ffttest_tikhonov(img_positive,img_negative,alpha=self.regularValue))
                    Pseudo_21 = extract_center_withShift(pseudo_ffttest_tikhonov(img_negative,img_positive,alpha=self.regularValue))

                Pseudo_tmp = np.stack([Pseudo_12,Pseudo_21],axis=-1)
            
                Pseudo_tmp = np.reshape(Pseudo_tmp,(32,32,2))
                single_out.append(Pseudo_tmp)
            outlist.append(single_out)
        
        return np.array(outlist)
