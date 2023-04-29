#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from rdmc.conformer_generation.task.molio import MolIOTask
from rdmc.conformer_generation.utils import get_binary, _software_available
from rdmc.external.inpwriter import (write_gaussian_opt,
                                     write_gaussian_freq,
                                     write_gaussian_irc)
from rdmc.external.logparser import GaussianLog

# Gaussian binary will be registered into software_available when calling get_binary
gbins = ['g16', 'g09', 'g03']
gaussian_binaries = {binname: get_binary(binname) for binname in gbins}

writer = {'opt': write_gaussian_opt,
          'freq': write_gaussian_freq,
          'irc': write_gaussian_irc}


class GaussianBaseTask(MolIOTask):
    """
    The base class to run Gaussian calculations.
    You have to have the Gaussian package installed to run this Task and its child Task.

    Args:
        method (str, optional): The method (level of theory) to be used. you can use the level of
                                theory available in Gaussian. Running gaussian with xTB is supported
                                in RDMC. Defaults to GFN2-xTB.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
        gaussian_binary (str, optional): The name of the gaussian binary, useful when there is
                                         multiple versions of Gaussian installed.
    """

    # Note, request_external_software is not used as usual (e.g., called by check_external_software)
    # as other tasks. This is only a list to show the user what software is required.
    request_external_software = gbins
    # The files to be saved
    # usually a gaussian input file (.gjf) and a log file (.log or .out)
    # For some calculations '.chk' is also needed.
    files = {'input_file': 'input.gjf',
             'log_file': 'input.log',
             'chk_file': 'input.chk',
             'output_file': 'input.out'}
    subtask_dir_name = 'gaussian'
    # This class by default uses GaussianLog to parse the log file.
    logparser = GaussianLog
    # The type of calculation to be performed (i.e., 'opt', 'freq', 'irc' for now)
    # This usually defined in the child class or the BaseCalculationTasks.
    calc_type = ''

    def check_external_software(self,
                                **kwargs,):
        """
        Check if Gaussian is installed and available in the environment variables.
        If the user specifies a Gaussian binary, use it.

        Args:
            gaussian_binary (str, optional): The name of the gaussian binary, useful
                                when there is multiple versions of Gaussian installed.
                                Defaults to None, no preference.

        Raises:
            RuntimeError: If no Gaussian installation is found.
        """
        # If the user specifies a Gaussian binary
        user_req_bin = _software_available.get(kwargs.get('gaussian_binary'))
        if user_req_bin:
            self.gaussian_binary = user_req_bin
        # Use the latest Gaussian binary found in the environment variables
        for binname in ['g16', 'g09', 'g03']:
            if _software_available.get(binname):
                self.gaussian_binaries = gaussian_binaries[binname]
                break
        else:
            raise RuntimeError('No Gaussian installation found.')

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  nprocs: int = 1,
                  memory: int = 1,):
        """
        Set up the Gaussian calculation. For the default implementation,
        it will set the method (level of theory), number of processors and memory,
        and save them as attributes.

        Args:
            method (str, optional): The level of theory that is available in Gaussian.
            nprocs (int, optional): The number of processors to use.
            memory (int, optional): Memory in GB used by Gaussian.

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
        Use the Gaussian writer to write the input file. The writer is defined in the
        rdmc.external.inpwriter module, and currently only supports 'opt', 'freq' and 'irc'
        for Gaussian calculations.

        Args:
            mol (RDKitMol): The molecule to be calculated.
            conf_id (int): The conformer ID of the molecule.
            **kwargs: Other arguments to be passed to the writer.

        return:
            str: The input file content.
        """
        return writer[self.calc_type](mo=mol,
                                      conf_id=conf_id,
                                      method=self.method,
                                      nprocs=self.nprocs,
                                      memory=self.memory,
                                      **kwargs)

    def get_execute_command(self,
                            subtask_id: int,
                            ) -> List[str]:
        """
        The command of executing the Gaussian binary. E.g., ['g16', 'input.gjf']
        This function task the subtask_id as an argument for locating the correct
        subtask directory.

        Args:
            subtask_id (int): The subtask ID of the calculation.

        Returns:
            List[str]: The command to be executed. E.g., ['g16', './input.gjf']
        """
        return [self.gaussian_binary, self.paths['input_file'][subtask_id]]
