#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.task import GaussianBaseTask


class GaussianOptimizer(IOOptimizer, GaussianBaseTask):
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

    subtask_dir_name = 'gaussian_opt'
    files = {'input_file': 'gaussian_opt.gjf',
             'log_file': 'gaussian_opt.log'}
    keep_files = ['gaussian_opt.gjf', 'gaussian_opt.log']

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the Gaussian log file and set them to the molecule.
        """
        log = self.logparser(self.paths['log_file'][subtask_id])
        # 1. Parse coordinates
        if log.success:
            mol.SetPositions(log.converged_geometries[-1], id=subtask_id)
            mol.GetConformer(subtask_id).SetIntProp('n_opt_cycles',
                                                    log.optstatus.shape[0] - 1)
        else:
            mol.keep_ids[subtask_id] = False
        # 2. Parse energy
        mol.energies[subtask_id] = \
                        log.get_scf_energies(converged=True,
                                             relative=False)[-1].item()
        mol.frequencies[subtask_id] = log.freqs
