#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task.molio import MolIOTask
from rdmc.conformer_generation.utils import get_binary, _software_available
from rdmc.external.inpwriter import (write_gaussian_opt,
                                     write_gaussian_freq,
                                     write_gaussian_irc)
from rdmc.external.logparser import GaussianLog

# Gaussian binary will be registered into software_available when calling get_binary
gaussian_binaries = {binname: get_binary(binname) for binname in ['g16', 'g09', 'g03']}

writer = {'opt': write_gaussian_opt,
          'freq': write_gaussian_freq,
          'irc': write_gaussian_irc}


class GaussianBaseTask(MolIOTask):
    """
    The base class to run Gaussian calculations.
    You have to have the Gaussian package installed to run this Task and its child Task.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
        gaussian_binary (str, optional): The name of the gaussian binary, useful when there is multiple versions of Gaussian installed.
    """

    label = 'GaussianBaseTask'
    # Note, request_external_software is not used for
    # check_external_software as other tasks
    request_external_software = ['g16', 'g09', 'g03']
    files = {'input_file': 'input.gjf',
             'log_file': 'input.log',
             'output_file': 'input.out'}
    # only save the input file and the log files
    keep_files = ['input.gjf', 'input.log']
    subtask_dir_name = 'gaussian'
    calc_type = ''
    logparser = GaussianLog

    def check_external_software(self,
                                **kwargs,):
        """
        Check if Gaussian is installed.
        If the user specifies a Gaussian binary, use it.
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
        Set up the Gaussian optimizer.
        """
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

    def input_writer(self,
                     mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs):
        """
        Use the Gaussian writer to write the input file.
        """
        return writer[self.calc_type](mo=mol,
                                      conf_id=conf_id,
                                      method=self.method,
                                      nprocs=self.nprocs,
                                      memory=self.memory,
                                      **kwargs)

    def write_input_file(self, **kwargs):
        """
        Use the default write_input_file function
        """
        return super().write_input_file(**kwargs)

    def get_execute_command(self, subtask_id: int) -> list:
        """
        The command of executing the Gaussian binary.
        E.g., ['g16', 'input.gjf']
        """
        return [self.gaussian_binary, self.paths['input_file'][subtask_id]]
