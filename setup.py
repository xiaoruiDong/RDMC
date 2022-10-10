#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RDMC", # Replace with your own username
    version="0.0.1",
    author="Xiaorui Dong",
    author_email="xiaorui@mit.com",
    description="An RDKit Wrapper for molecule and conformer operation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaoruiDong/RDMC",
    packages=find_packages(),
    install_requires=['numpy', 'py3Dmol'],  # you install rdkit and openbabel from environment.yml
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    license = "MIT License",
    python_requires='>=3.6',
)
