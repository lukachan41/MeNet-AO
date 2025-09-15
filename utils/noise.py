import numpy as np
    
def add_poisson_gaussian_noise_np(img, level=1000.0):
    img = img.astype(np.float32)
    if np.max(img) == 0.0:
        poisson = np.random.poisson(lam=0.0, size=img.shape)
    else:
        poisson = np.random.poisson(lam=img * level)

    gaussian = np.random.normal(loc=100.0, scale=4.5, size=img.shape)

    img_noised = poisson + gaussian

    if np.max(img_noised) - np.min(img_noised) == 0.0:
        raise ValueError("Noised image has no dynamic range.")
    img_noised = (img_noised - np.min(img_noised)) / (np.max(img_noised) - np.min(img_noised))

    return img_noised