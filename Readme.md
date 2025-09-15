# MeNet-AO

MeNet-AO is a Python package for fast optical aberration estimation from fluorescence images using Physics-informed multi-encoder trained on synthetic images or experimental acquired images.

This repository is developed based on the paper —"Physics-informed multi-encoder adaptive optics enables rapid aberration correction for intravital microscopy of deep complex tissue" .

## Environment & Configuration

- Windows 10
- Python 3.9
- Tensorflow 2.6.0 & keras 2.6.0
- GPU: GeForce RTX 3090Ti
- Create a virtual environment and install dependencies from requirements.txt

## File structure

- ./Data/img is the default path to save aberration-free images
- ./Dataset/ is the default save path for training data and testing data
- ./Demo/  contains notebooks that can be run directly to reproduce the results from the paper.
- ./model/  includes the declaration of the MeNet-AO model and the organization of data
- ./training/  defines data generation and training procedures, enabling model training from scratch.
- ./utils/  is the tool package of MeNet-AO,which includes the PSF generator, aberration-related feature extraction, and other supporting tools.

## Dataset

We provide several datasets (including both simulation-generated and experimentally acquired) for running **MeNet-AO**. Each dataset is designed to correspond to the demos described below.

The dataset can be download from [Data for MeNet-AO demo](https://doi.org/10.5281/zenodo.17118222) 

## Demo

We provide several demo notebooks to reproduce and illustrate the results presented in the paper.

**Demo_1:**  Initial_aberrations: $$Z_{5} / Z_{6} /Z_{7} /Z_{8} /Z_{9} /Z_{10} /Z_{11}$$ ; Modulated_aberration: **$$Z_{5}$$** (Simulation). This demo reproduces the results corresponding to *Figure 1b*, where **$$Z_{5}$$ (Oblique astigmatism)** is shown to provide superior multi-modal predictability when used as the modulated aberration.

**Demo_2:** Initial_aberrations: $$Z_{5} / Z_{6} /Z_{7} /Z_{8} /Z_{9} /Z_{10} /Z_{11}$$ ; Modulated_aberration: $$Z_{7}$$ (Simulation). This demo reproduces the results corresponding to *Figure 1b*, where $$Z_{7}$$ **(Vertical Coma)** is shown to provide **limited** multi-modal predictability when used as the modulated aberration. 

**Demo_3:** Initial_aberrations: $$Z_{5} , Z_{6} , Z_{7} , Z_{8} , Z_{9} , Z_{10} ,Z_{11}$$ ; Modulated_aberration: $$Z_{5} , Z_{6} ,Z_{11}$$ (Simulation). This demo is to demonstrate the superior performance for estimate all primary aberration of MeNet-AO. 

**Demo_4:**  Initial_aberrations: $$Z_{5} , Z_{6} , Z_{7} , Z_{8} , Z_{9} , Z_{10} ,Z_{11}$$ ; Modulated_aberration: $$Z_{5} , Z_{6} ,Z_{11}$$ (Experimental). This demo demonstrates the robust and practical performance of MeNet-AO in estimating all primary aberrations under noise conditions and varying aberration amplitudes.

## Train a new model

**DataGeneration:**  You can generate a new dataset by running **`DataGenerator.py`**. In the script, you can configure the initial aberration, modulation modes, and other parameters as needed. After execution, the dataset along with the corresponding parameters will be saved to the path you specify.

**Model Training:** Run **`Model_training.ipynb`** following the instructions provided in the notebook.
 Make sure to set the `file_path` to the same location specified in **`DataGenerator.py`**, so that the algorithm can properly load the dataset.  The name and save path of the trained model can be customized as desired.



