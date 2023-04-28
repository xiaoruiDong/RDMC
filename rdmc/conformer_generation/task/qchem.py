#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task.molio import MolIOTask
from rdmc.conformer_generation.utils import get_binary
from rdmc.external.inpwriter import (write_qchem_opt,
                                     write_qchem_freq,
                                     write_qchem_irc)
from rdmc.external.logparser import QChemLog

qchem_binary = get_binary('qchem')
writer = {'opt': write_qchem_opt,
          'freq': write_qchem_freq,
          'irc': write_qchem_irc}


class QChemBaseTask(MolIOTask):
    """
    The class to optimize geometries using the algorithm built in QChem.
    You have to have the QChem package installed to run this optimizer.

    Args:
        method (str, optional): The method available in QChem to be used for QChem calculation
                                Defaults to wb97x-d3.
        basis (str, optional):  The basis set to use. Defaults to def2-svp.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
    """

    request_external_software = ['qchem']
    files = {'input_file': 'input.qcin',
             'log_file': 'input.log',
             'output_file': 'input.out'}
    keep_files = ['*']
    subtask_dir_name = 'qchem'
    calc_type = ''
    logparser = QChemLog

    def task_prep(self,
                  method: str = "wb97x-d3",
                  basis: str = 'def2-svp',
                  nprocs: int = 1,
                  **kwargs,
                  ):
        self.method = method
        self.basis = basis
        self.nprocs = nprocs

    def input_writer(self,
                     mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs):
        """
        Use the QChem writer to write the input file.
        """
        return writer[self.calc_type](mol=mol,
                                      conf_id=conf_id,
                                      method=self.method,
                                      basis=self.basis,
                                      nprocs=self.nprocs,
                                      **kwargs)

    def get_execute_command(self, subtask_id: int) -> list:
        """
        The command of executing the QChem binary.
        E.g., ['qchem', 'input.qcin']
        """
        return [qchem_binary, self.paths['input_file'][subtask_id]]
