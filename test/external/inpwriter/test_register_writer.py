#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the registration of external input writers.
"""

import pytest

from rdmc.external.inpwriter._register import (_registered_qm_writers,
                                               register_qm_writer,
                                               get_qm_writer)


def test_register_qm_writer():
    """
    Test the base function of registration of external QM input writers.
    """
    def test_writer(x: str):
        return x + '\ntest\n'
    register_qm_writer('Test', 'Test', test_writer)

    with pytest.raises(KeyError):
        _registered_qm_writers['Test']['Test']
    assert _registered_qm_writers['test']['test'] == test_writer
    # Post process
    del _registered_qm_writers['test']


@pytest.mark.parametrize('software, job_type, writer',
                         [('gaussian', 'opt', 'write_gaussian_opt'),
                          ('g03', 'freq', 'write_gaussian_freq'),
                          ('g09', 'IRC', 'write_gaussian_irc'),
                          ('g16', 'GSM', 'write_gaussian_gsm'),
                          ('orca', 'Frequency', 'write_orca_freq'),
                          ('orca', 'Growing String Method', 'write_orca_gsm'),
                          ('qchem', 'optimization', 'write_qchem_opt'),
                          ])
def test_get_qm_writer(software: str, job_type: str, writer: str):
    """
    Test the function of getting the external QM input writer.
    """
    writer = get_qm_writer(software, job_type)
    assert callable(writer)
    assert isinstance(writer.__name__, str)


def test_get_qm_writer_fail():
    """
    Test the failure case of getting the external QM input writer.
    """
    with pytest.raises(ValueError):
        get_qm_writer('invalid_software', 'invalid_job_type')
    with pytest.raises(ValueError):
        get_qm_writer('gaussian', 'invalid_job_type')
