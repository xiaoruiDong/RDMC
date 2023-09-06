#!/usr/bin/env python3
#-*- coding: utf-8 -*-

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
                      # RDKit and OpenBabel are canceled out to avoid duplicated
                      # installation by PyPI
                      # OpenBabel also has installation issues at least on Mchip Mac
                      # 'rdkit>=2021.03.1',
                      # 'openbabel',
                      'networkx',
                      'py3Dmol',
                      'ase',
                      'matplotlib',
                      'cclib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    keywords="chemistry, RDKit, molecule, conformer, reaction",
    license = "MIT License",
    python_requires='>=3.6',
    platforms=["Any."],
)
