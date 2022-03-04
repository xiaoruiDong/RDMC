# ts_egnn
transition state guess generation with equivariant neural networks

## Requirements

* python (version==3.7)
* rdkit (version==2021.09.3)
* pytorch (version==1.8.1)
* torch-sparse (version==0.6.12)
* torch-scatter (version==2.0.9)
* pytorch-geometric (version==1.7.1)
* pytorch-lightning (version==1.5.7)
* egnn_pytorch (version==0.2.6)
* rdmc (version==0.0.1)
* rmsd (version==1.4)

## Installation
Installation of some of these packages can be a headache. I am still trying to figure out the best way to install torch
and torch-geometric, but here is the suggested way:

### Create the environment
```
conda create -n ts_egnn python=3.7
```

### Install pytorch
```
# CUDA 10.2
conda install pytorch==1.8.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# CPU only
conda install pytorch==1.8.1 cpuonly -c pytorch
```

### Install pytorch-geometric and associated packages
```
IDX={cu102, cu113, cpu} (depending on cuda installation)
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.1+${IDX}.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+${IDX}.html
pip install torch-geometric==1.7.1
```

### Install other required packages which should be smooth
```
conda install -c rdkit rdkit
pip install egnn-pytorch
pip install pytorch-lightning
pip install rmsd
```

### Clone RDMC and install
```
git clone https://github.com/xiaoruiDong/RDMC
cd RDMC
pip install -e .
cd ..
```

### Finally, clone this repo
```
git clone https://github.com/PattanaikL/ts_egnn
```

### I used Neptune to keep track of experiments, and I thought it was useful to keep my experiments organized (optional, only for training)
```
pip install neptune-client
```
