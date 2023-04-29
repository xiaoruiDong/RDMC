#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from rdmc.conformer_generation.task.molio import MolIOTask
from rdmc.conformer_generation.utils import get_binary
from rdmc.external.inpwriter import (write_orca_opt,
                                     write_orca_freq,
                                     write_orca_irc)
from rdmc.external.logparser import ORCALog

orca_binary = get_binary('orca')
writer = {'opt': write_orca_opt,
          'freq': write_orca_freq,
          'irc': write_orca_irc}


class ORCABaseTask(MolIOTask):
    """
    The base class to run ORCA calculations.
    You have to have the ORCA package installed to run this optimizer and its child Task

    Args:
        method (str, optional): The method (level of theory) available in ORCA to be used.
                                If you want to use XTB methods, you need to put the xtb binary
                                into the Orca directory. Defaults to XTB2.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): The memory to use in GB. Defaults to 1.
    """

    request_external_software = ['orca']
    # The files to be saved
    # usually an ORCA input file (.inp) and a log file (.log)
    # For opt calculations 'input.out' is also useful..
    files = {'input_file': 'input.inp',
             'log_file': 'input.log',
             'output_file': 'input.out',
             'output_xyz': 'input.xyz'}
    subtask_dir_name = 'orca'
    # This class by default uses ORCALog to parse the log file.
    logparser = ORCALog
    # The type of calculation to be performed (i.e., 'opt', 'freq', 'irc' for now)
    # This usually defined in the child class or the BaseCalculationTasks.
    calc_type = ''

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  nprocs: int = 1,
                  memory: int = 1,
                  **kwargs,
                  ):
        """
        Set up the ORCA calculation. For the default implementation,
        it will set the method (level of theory), number of processors and memory,
        and save them as attributes.

        Args:
            method (str, optional): The level of theory that is available in ORCA.
            nprocs (int, optional): The number of processors to use.
            memory (int, optional): Memory in GB used by ORCA.

        For developers: You can treat this function as __init__. without the need to
        call super().__init__().
        """
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

    def input_writer(self,
                     mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs,
                     ) -> str:
        """
        Use the ORCA writer to write the input file. The writer is defined in the
        rdmc.external.inpwriter module, and currently only supports 'opt', 'freq' and 'irc'
        for ORCA calculations.

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
                                      nprocs=self.nprocs,
                                      memory=self.memory,
                                      **kwargs)

    def get_execute_command(self,
                            subtask_id: int,
                            ) -> List[str]:
        """
        The command of executing the ORCA binary. E.g., ['orca', 'input.inp']
        This function task the subtask_id as an argument for locating the correct
        subtask directory.

        Args:
            subtask_id (int): The subtask ID of the calculation.

        Returns:
            List[str]: The command to be executed. E.g., ['orca', './input.inp']
        """
        return [orca_binary, self.paths['input_file'][subtask_id]]
