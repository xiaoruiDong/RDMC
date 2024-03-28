RDMC currently supports several machine-learning based tasks

### Embedders
- GeoMol Embedder
- Torsional-Diffusion Embedder

### TS Guessers
- TS-EGNN
- TS-GCN

### TS Verifier
- TS-Screener

As well as several different QM software for optimization and verfication
- xTB
- Gaussian
- ORCA
- QChem

## Setup Python Environment

Here is an example of getting workable environment
```
# Create Environment
# - you can change the name from ts_ml to other names
# - you may change the python version from 3.11 to 3.8 - 3.12
mamba create -n ts_ml python=3.11
mamba activate ts_ml

# Install Pytorch, PyG and Pytorch-lightning first
# This is important to be installed first to lock
# the most important dependencies

# You can change the PyTorch, PyG, and CUDA versoin
# Depending on your system setup and preferences
# Note: to have TS-EGNN working, the following versions are recommended
mamba install pytorch=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install pyg=2.4.0 -c pyg -y  # pyg 2.3 should work as well
mamba install pytorch-lightning -y
mamba install einops -y

# After every step since now, carefully check PyTorch's version and channel, it is not expected to be changed

# Install GeoMol
# Extending PyG. Modify the torch and cuda version based on your previous installation
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
mamba install pot yaml pyyaml -y
git clone https://github.com/xiaoruiDong/GeoMol
cd GeoMol
pip install -e .
cd ../

# TS-EGNN, TS-GCN, TS-Screener
# Private Repo as of Apr 2024, contact Xiaorui for access
git clone https://github.com/xiaoruiDong/TS-ML
mamba install seaborn neptune -y
cd TS-ML
pip install -e .
cd ..

# Torsional-Diffusion
git clone https://github.com/xiaoruiDong/torsional-diffusion
mamba install spyrmsd pyyaml -y
pip install e3nn
cd torsional-diffusion
# download the workdir from https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7
# you can use gdown and https://sites.google.com/site/gdocs2direct/ to transform url to download to headless server
cd ..

# Install XTB
# Only xtb <= 6.4.1 works properly with Gaussian EIn
mamba install xtb=6.4.1 crest

# Install RDMC
git clone https://github.com/xiaoruiDong/RDMC
mamba install rdkit openbabel matplotlib py3dmol ase cclib -y
bash setup.sh

# Install Jupyter Notebook for visualization and run examples
mamba install jupyter -y
```