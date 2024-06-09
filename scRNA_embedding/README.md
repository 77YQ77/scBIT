

## Introduction

These codes are for the first stage of the scBIT framework. 
The purpose of these codes is to learn the representation for the generated gene-gene interaction subgraph.

## Requirements

This network framework is mainly implemented through PyTorch and PyTorch Geometric.

```
* python                3.9.0
* torch                 1.10.1+cu113
* torch_geometric       2.4.0
```
You may find all the required packages in the *requirements.txt file.*

## Data Preparation
Datasets should be downloaded into './datasets/'. Hyper-parameters can be changed in 'Configures.py'
Each data file should contain two parts:

-One is the gene expression matrix file, its rows represent genes and columns represent cells. The naming format is as follows: ```{species}_{tissue}{id}_data.csv```

-The other is the file of celltype. The naming format is: ```{species}_{tissue}{id}_celltype.csv```

## Usage

### Command Line Parameters:
```--clst``` is the clst factor, you can change it in 'Configures.py', default=0
```--sep``` is the sep factor, you can change it in 'Configures.py', default=0
```--id```  is the id of data file, default=1
```dataset``` is name of the data set, change this param for using different dataset

### Train model:
You can simply run train process with default args by:
```
python -m models.train_gnns
```

### Get prototype vectors:
The prototype vectors have been saved in the checkpoint model
Load the saved model and use ``` model['net']['model.prototype_vectors'] ``` to get them














