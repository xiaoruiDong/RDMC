#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RDMC",
    version="0.1.0",
    author="Xiaorui Dong, Lagnajit Pattanaik, Shih-Cheng Li, Kevin Spiekermann, Hao-Wei Pang, and William H. Green",
    author_email="xiaorui@mit.com",
    description="A light-weight software package with expertise in handling Reaction Data and Molecular (including transitions states) Conformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaoruiDong/RDMC",
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'rdkit>=2021.03.1',
                      'openbabel-wheel>=3.1.1',
                      'networkx',
                      'py3Dmol',
                      'ase',
                      'matplotlib',
                      'cclib',
                      'ipywidgets',  # view molecules (not required to specify when using conda/mamba)
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    keywords="chemistry, RDKit, molecule, conformer, reaction, cheminformatics",
    license="MIT License",
    python_requires='>=3.7',
    platforms=["Any."],
)
