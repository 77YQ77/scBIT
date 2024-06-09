## Introduction
This is the official implementation of the second and third stages of scBIT. 

The second stage builds pre-trained brain ROI-level encoders to learn ROI-level embeddings from fMRI samples, which are used to calculate the cross-modal attention values using the calculated gene subgraph embeddings. 
The third stage adaptively fuses the cross-modal attention and the ROI-level embeddings for Alzheimer's diagnosis.

## Requirements

This network framework is implemented through PyTorch.

```
* python                3.8.1
* torch                 1.8.0+cu111
```
You may find all the required packages in the *requirements.txt file.*

### Data Preparation
fMRI datasets and labels can be downloaded from [ADNI](https://adni.loni.usc.edu/). The datasets are further processed by the AAL atlas and normalized, named 'fmri_data.npy' and stored into './data/.' The gene subgraph embeddings and different similarities, calculated in **_scRNA embedding_**, should also be converted into 'npy' format and included in './data/.'

### Usage
Using the ```main_fusion.py``` to train and test the model on your own dataset. The proposed network is defined in the ```model.py```. It can be easily edited and embedded in your own code.


