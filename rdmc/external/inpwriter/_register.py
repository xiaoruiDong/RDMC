#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module registers the external input writers.
"""

from typing import Callable
from collections import defaultdict

_registered_qm_writers = defaultdict(dict)


def register_qm_writer(software: str,
                       job_type: str,
                       writer: Callable,
                       ):
    """
    Register a QM input writer.

    Args:
        software (str): The software to be registered.
        job_type (str): The job type to be registered.
        writer (callable): The writer to be registered.
    """
    _registered_qm_writers[software.lower()][job_type.lower()] = writer


def get_qm_writer(software: str,
                  job_type: str,
                  ) -> Callable:
    """
    Get the QM input writer.

    Args:
        software (str): Options: ``"gaussian"``, ``"orca"``, ``qchem``.
        job_type (str): Options: ``"opt"``, ``"freq"``, ``"irc"``, ``"gsm"``.

    Returns:
        Callable: The QM input writer function.
    """
    if software.lower() in ['gaussian', 'g03', 'g09', 'g16']:
        software = 'gaussian'
    if job_type.lower() == 'optimization':
        job_type = 'opt'
    elif job_type.lower() == 'frequency':
        job_type = 'freq'
    elif job_type.lower() in ['growing string', 'growing string method']:
        job_type = 'gsm'
    try:
        return _registered_qm_writers[software.lower()][job_type.lower()]
    except KeyError:
        if _registered_qm_writers.get(software.lower(), None) is None:
            raise ValueError(f"Software {software} is not available.")
        else:
            raise ValueError(f"Job type {job_type} is not available for software {software}.")
