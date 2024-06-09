## Introduction
This is the official implementation of the second and third stages of scBIT. 
The second stage builds pre-trained brain ROI-level encoders to learn ROI-level embeddings from fMRI samples, which are used to calculate the cross-modal attention values using the calculated gene subgraph embeddings. 
The third stage adaptively fuses the cross-modal attention and the ROI-level embeddings for Alzheimer's diagnosis.

### Data Preparation
Datasets and labels can be downloaded from [ADNI](https://adni.loni.usc.edu/). The datasets are further processed by the AAL atlas and normalized, named 'fmri_data_norm. npy' and stored into './data/.' The gene subgraph embeddings, calculated in 
Four different similarities should also be converted into 'npy' format and included in './data/.'
