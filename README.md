![RDMC Logo](docs/source/_static/RDMC_icon.svg)
# Reaction Data and Molecular Conformer

[![Documentation](https://github.com/xiaoruiDong/RDMC/actions/workflows/build_docs.yaml/badge.svg)](https://xiaoruidong.github.io/RDMC/)
[![CI](https://github.com/xiaoruiDong/RDMC/actions/workflows/ci.yaml/badge.svg)](https://github.com/xiaoruiDong/RDMC/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/xiaoruiDong/RDMC/graph/badge.svg?token=5LT5A35783)](https://codecov.io/gh/xiaoruiDong/RDMC)
[![Anaconda Cloud](https://img.shields.io/conda/v/xiaoruidong/rdmc)](https://anaconda.org/xiaoruidong/rdmc)
[![PyPI](https://img.shields.io/pypi/v/rdmc)](https://pypi.org/project/rdmc/)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

A light-weight software package with expertise in handling Reaction Data and Molecular (including transitions states) Conformers.

The package can be easily installed with conda or mamba

```
conda install -c xiaoruidong rdmc
```

For detailed APIs, please check the [documentation](https://xiaoruidong.github.io/RDMC/).

## Demos
Feel free to check demos in the `ipython/`, some of them are also available on the Google Colab:
- [Generate Atom Map for Reactions](https://colab.research.google.com/drive/19opX3Sr4R24o9n8f1o4LMSqlVIwN83xk?usp=sharing)
- [Handle molecule from/to XYZ](https://colab.research.google.com/drive/1QbmdvUMQqByPBDQVW7xTlp2rXg9EJ2_J?usp=sharing)
- [Parse QM Results](https://colab.research.google.com/drive/1JnTzETOGE3R3Q_foOLsnFgeN883J36dl?usp=sharing)

## Requirements
* python
* numpy
* scipy
* rdkit
* openbabel
* py3dmol
* ase
* networkx
* matplotlib
* cclib
