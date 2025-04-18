#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides methods that can directly apply to RDKit Mol/RWMol.
"""
import inspect
import pathlib

repo_dir = pathlib.Path(__file__).absolute().parent.parent


def filter_kwargs(func, kwargs: dict):

    parameters = inspect.signature(func).parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in parameters}

    return valid_kwargs
