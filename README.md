# Advancing brain imaging transcriptomics to single-cell level for Alzheimer's disease diagnosis
## An Alzheimer's disease diagnosis method based on the adaptive matching of fMRI samples and snRNA data.

This repository provides the Python implementation of advancing brain imaging transcriptomics to single-cell level for Alzheimer's disease diagnosis (scBIT). 

The codes are divided into two parts:

**1. _scRNA embedding_**

**2. _AD diagnosis_**

The users should first utilize codes for **_scRNA embedding_** to generate the gene subgraph embeddings for snRNA data. Subsequently, the users can leverage the codes for **_AD diagnosis_**, which learns the feature embeddings from fMRI samples and can adaptively fuse the knowledge from fMRI samples and snRNA data to improve the diagnostic performance of Alzheimer's disease.

### Installation
Since these two stages are conducted in different environments, the users are encouraged to create different virtual environments and run the following commands to install the required packages.
* for installing reliance of scRNA_embedding:
```
pip install -r scRNA_embedding/requirements.txt
```
* the command for installing reliance of AD_diagnosis:
```
pip install -r scRNA_embedding/requirements.txt
```

### Usage
You may find the introduction to run the codes for each stage once you open the corresponding folders.
