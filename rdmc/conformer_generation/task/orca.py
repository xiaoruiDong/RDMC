#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    The class to optimize geometries using the algorithm built in ORCA.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method available in ORCA to be used for TS optimization.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory. Defaults to XTB2.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): The memory to use in GB. Defaults to 1.
    """

    request_external_software = ['orca']
    files = {'input_file': 'input.inp',
             'log_file': 'input.log',
             'output_file': 'input.out',
             'output_xyz': 'input.xyz'}
    keep_files = ['*']
    subtask_dir_name = 'orca'
    calc_type = ''
    logparser = ORCALog

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  nprocs: int = 1,
                  memory: int = 1,
                  **kwargs,
                  ):
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

    def input_writer(self,
                     mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs):
        """
        Use the ORCA writer to write the input file.
        """
        return writer[self.calc_type](mol=mol,
                                      conf_id=conf_id,
                                      method=self.method,
                                      nprocs=self.nprocs,
                                      memory=self.memory,
                                      **kwargs)

    def get_execute_command(self, subtask_id: int) -> list:
        """
        The command of executing the ORCA binary.
        E.g., ['orca', 'input.inp']
        """
        return [orca_binary, self.paths['input_file'][subtask_id]]
