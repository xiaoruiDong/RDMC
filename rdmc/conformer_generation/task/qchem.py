#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

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
    The base class to run QChem calculations.
    You have to have the QChem package installed to run this optimizer and its child Task.
    Note, the argument `method` is different in QChem than in ORCA/Gaussian, which shouldn't
    contain basis set information in QChem Tasks, as there is a specific arguement for it.
    The is no memory argument as QChem will handle it.

    Args:
        method (str, optional): The method (level of theory) available in QChem to be used.
                                defaults to wb97x-d3.
        basis (str, optional): The basis set to use. Defaults to def2-svp.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
    """

    request_external_software = ['qchem']
    # The files to be saved
    # usually an QChem input file (.qcin) and a log file (.log)
    files = {'input_file': 'input.qcin',
             'log_file': 'input.log',
             'output_file': 'input.out'}
    subtask_dir_name = 'qchem'
    # This class by default uses QChemLog to parse the log file.
    logparser = QChemLog
    # The type of calculation to be performed (i.e., 'opt', 'freq', 'irc' for now)
    # This usually defined in the child class or the BaseCalculationTasks.
    calc_type = ''

    def task_prep(self,
                  method: str = "wb97x-d3",
                  basis: str = 'def2-svp',
                  nprocs: int = 1,
                  **kwargs,
                  ):
        """
        Set up the QChem calculation. For the default implementation,
        it will set the method, basis set, number of processors and memory,
        and save them as attributes.

        Args:
            method (str, optional): The method that is available in QChem.
            basis (str ,optional): The basis set that is available in QChem
            nprocs (int, optional): The number of processors to use.

        For developers: You can treat this function as __init__. without the need to
        call super().__init__().
        """
        self.method = method
        self.basis = basis
        self.nprocs = nprocs

    def input_writer(self,
                     mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs,
                     ) -> str:
        """
        Use the QChem writer to write the input file. The writer is defined in the
        rdmc.external.inpwriter module, and currently only supports 'opt', 'freq' and 'irc'
        for QChem calculations.

        Args:
            mol (RDKitMol): The molecule to be calculated.
            conf_id (int): The conformer ID of the molecule.
            **kwargs: Other arguments to be passed to the writer.

        return:
            str: The input file content.
        """
        return writer[self.calc_type](mol=mol,
                                      conf_id=conf_id,
                                      method=self.method,
                                      basis=self.basis,
                                      nprocs=self.nprocs,
                                      **kwargs)

    def get_execute_command(self,
                            subtask_id: int,
                            ) -> List[str]:
        """
        The command of executing the QChem binary. E.g., ['qchem', 'input.qcin']
        This function task the subtask_id as an argument for locating the correct
        subtask directory.

        Args:
            subtask_id (int): The subtask ID of the calculation.

        Returns:
            List[str]: The command to be executed. E.g., ['qchem', './input.qcin']
        """
        return [qchem_binary, self.paths['input_file'][subtask_id]]
