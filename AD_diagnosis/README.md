## Introduction
This is the official implementation of the second and third stages of scBIT. 
The second stage builds pre-trained brain ROI-level encoders to learn ROI-level embeddings from fMRI samples, which are used to calculate the cross-modal attention values using the calculated gene subgraph embeddings. 
The third stage adaptively fuses the cross-modal attention and the ROI-level embeddings for Alzheimer's diagnosis.

## Requirements

This network framework is mainly implemented through PyTorch.

```
* python                3.8.1
* torch                 1.8.0+cu111
```
You may find all the required packages in the  ```requirements.txt```.

### Data Preparation
fMRI datasets and labels can be downloaded from [ADNI](https://adni.loni.usc.edu/). Please follow the relevant regulations to download from the website. 
The datasets are further processed by the AAL atlas and normalized, named 'fmri_data.npy' and stored into './data/.' The gene subgraph embeddings and different similarities, calculated in **_scRNA embedding_**, should also be converted into 'npy' format and included in './data/.'

Four different similarities between snRNA and fMRI samples are calculated as follows:

- **_Gender Similarity_**: Gender similarity is calculated by assigning a value of 1 for males and -1 for females, then computing the difference between brain region patients and single-cell patients, followed by taking the absolute value.
- **_Age Similarity_**: Age similarity is calculated by capping ages over 90 at 90, then computing the difference between brain region patients and single-cell patients, followed by taking the absolute value.
- **_Disease Condition Similarity_**: Disease condition similarity is calculated as follows: brain region conditions are scored as Normal=0, Early=0.25, Middle=0.5, Late=0.75, AD=1; single-cell conditions are scored as Normal=0, Low=0.25, Medium=0.5, High=0.75; the difference between brain region patients and single-cell patients is then computed, followed by taking the absolute value.
- **_Genetic Information Similarity_**: Genetic similarity is calculated using SNP data with preprocessing steps including filtering for missing rates, Hardy-Weinberg equilibrium testing, minor allele frequency filtering, and linkage disequilibrium pruning. Subsequently, we use the AUCell algorithm to obtain matrices representing the relationship of brain regions and single-cell data with KEGG gene pathways, and then compute the cosine similarity.


### Usage
Using the ```main_fusion.py``` to train and test the model with default hyperparameters on ADNI dataset. The proposed network is defined in the ```model.py```. It can be easily edited and embedded in your own code.


