#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.task import GaussianBaseTask


class GaussianOptimizer(GaussianBaseTask, IOOptimizer):
    """
    The class to optimize geometries using the algorithm built in Gaussian.
    You have to have the Gaussian package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
        gaussian_binary (str, optional): The name of the gaussian binary, useful when there is multiple versions of Gaussian installed.
                                         Defaults to the latest version found in the environment variables.
    """

    label = 'GaussianOptimizer'
    subtask_dir_name = 'gaussian_opt'
    files = {'input_file': 'gaussian.gjf',
             'log_file': 'input.log'}
    keep_files = ['gaussian.gjf', 'input.log']
    create_mol_flag = True
    init_attrs = {'energies': np.nan, 'frequencies': None}
    calc_type = 'opt'

    def save_data(self, **kwargs):
        """
        Save the data.
        """
        super(GaussianBaseTask, self).save_data(**kwargs)  # from IOOptimizer

    def post_run(self, **kwargs):
        """
        Setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super(GaussianBaseTask, self).post_run(**kwargs)  # from IOOptimizer

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the xTB output file and set them to the molecule.
        """
        log = self.logparser(self.paths['log_file'][subtask_id])
        # 1. Parse coordinates
        if log.success:
            mol.SetPositions(log.converged_geometries[-1], id=subtask_id)
        else:
            mol.keep_ids[subtask_id] = False
        # 2. Parse energy
        mol.energies[subtask_id] = \
                        log.get_scf_energies(converged=True,
                                             relative=False)[-1].item()
        mol.frequencies[subtask_id] = log.freqs
