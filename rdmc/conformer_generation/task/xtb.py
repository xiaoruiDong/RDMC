#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

from rdmc.conformer_generation.task import MolIOTask
from rdmc.conformer_generation.utils import _software_available
from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
from rdmc.external.xtb_tools.utils import XTB_BINARY

_software_available['xtb'] = osp.isfile(XTB_BINARY)


class XTBBaseTask(MolIOTask):

    """
    The base class to run xTB calculations.
    You have to have the xTB package installed to run this Task and its child Task.

    Args:
        method (str, optional): The method to be used for xTB calculation. Defaults to GFN2-xTB.
    """

    label = 'XTBBaseTask'
    request_external_software = ['xtb']
    subtask_dir_name = 'xtb'
    calc_type = ''

    def task_prep(self,
                  method: str = "gfn2",
                  **kwargs,):
        """
        Prepare the task.
        """
        self.method = method

    def write_input_file(self, **kwargs):
        """
        Since run_xtb_calc is used, no input file is needed.
        """
        return

    def runner(self,
               mol: 'RDKitMol',
               subtask_id: int,
               mult: int,
               **kwargs):
        """
        Run the xTB calculation.
        """
        return run_xtb_calc(mol,
                            conf_id=subtask_id,
                            uhf=mult - 1,
                            job=self.calc_type,
                            method=self.method,
                            save_dir=self.paths['subtask_dir'][subtask_id],
                            **kwargs)
