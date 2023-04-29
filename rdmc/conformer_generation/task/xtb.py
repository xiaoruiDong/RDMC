#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

from rdmc.conformer_generation.task.molio import MolIOTask
from rdmc.conformer_generation.utils import _software_available
from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
from rdmc.external.xtb_tools.utils import XTB_BINARY

_software_available['xtb'] = osp.isfile(XTB_BINARY)
calc_type_dict = {
    'opt': '--opt',
    'freq': '--hess',
}


class XTBBaseTask(MolIOTask):

    """
    The base class to run xTB calculations.
    You have to have the xTB package installed to run this Task and its child Task.

    Args:
        method (str, optional): The method to be used for xTB calculation. Defaults to GFN2-xTB.
        level (str, optional): The criteria level used in optimization. Only valid for optimization tasks.
                               Defaults to 'tight'
    """

    request_external_software = ['xtb']
    subtask_dir_name = 'xtb'
    # The type of calculation to be performed (i.e., 'opt' and 'freq' for now)
    # This usually defined in the child class or the BaseCalculationTasks.
    calc_type = ''

    def task_prep(self,
                  method: str = "gfn2-xtb",
                  level: str = "tight",
                  **kwargs,):
        """
        Set up the xTB calculation. For the default implementation,
        it will set the method (level of theory) and optimization criteria level,
        and save them as attributes.

        Args:
            method (str, optional): The level of theory that is available in xTB.
                                    Defaults to 'gfn2-xtb'.
            level (str, optional): The criteria used in optimization. Defaults to 'tight'

        For developers: You can treat this function as __init__. without the need to
        call super().__init__().
        """
        self.method = method
        self.level = level

    def write_input_file(self,
                         **kwargs):
        """
        Skip the input file step as it is implemented in the execution implementation.

        For developers: Since the runner (run_xtb_calc) will write input file, this step
        can be skipped.
        """
        return

    def runner(self,
               mol: 'RDKitMol',
               subtask_id: int,
               **kwargs):
        """
        The runner of each subtask. In the xTB implementation, it is actually a wrapper
        of `run_xtb_calc` defined in RDMC.external.xtb_tools.

        For developers, as a note kwargs should have keys of charge and mult, as they are
        assigned by the second step of `run` in `MolIOTask`. Be careful if you need to
        reimplement `run` or `runner`.
        """
        return run_xtb_calc(mol,
                            conf_id=subtask_id,
                            uhf=kwargs['mult'] - 1,
                            job=calc_type_dict[self.calc_type],
                            method=self.method,
                            level=self.level,
                            save_dir=self.work_dir,
                            **kwargs)  # charge is stored inside kwargs
