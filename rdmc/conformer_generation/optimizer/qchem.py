#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.task import QChemBaseTask


class QChemOptimizer(QChemBaseTask, IOOptimizer):
    """
    The class to optimize geometries using the algorithm built in QChem.
    You have to have the QChem package installed and run `source qcenv.sh` to run this optimizer.

    Args:
        method (str, optional): The method available in ORCA to be used for TS optimization.
                                Defaults to wb97x-d3.
        basis (str, optional): The basis set to use. Defaults to def2-svp.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
    """

    label = 'QChemOptimizer'
    subtask_dir_name = 'qchem_opt'
    files = {'input_file': 'qchem_opt.qcin',
             'log_file': 'qchem_opt.log',
             'output_file': 'qchem_opt.out'}
    keep_files = ['*']
    create_mol_flag = True,
    init_attrs = {'energies': np.nan, 'frequencies': None}
    calc_type = 'opt'

    def save_data(self, **kwargs):
        """
        Save the data.
        """
        super(QChemBaseTask, self).save_data(**kwargs)  # from IOOptimizer

    def post_run(self, **kwargs):
        """
        Setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super(QChemBaseTask, self).post_run(**kwargs)  # from IOOptimizer

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the QChem log file and set them to the molecule.
        """
        log = self.logparser(self.paths['log_file'][subtask_id])
        # 1. Parse coordinates
        if log.success:
            mol.SetPositions(log.converged_geometries[-1],
                             id=subtask_id)
            mol.GetConformer(subtask_id).SetIntProp('n_opt_cycles',
                                                    log.optstatus.shape[0] - 1)
        else:
            mol.keep_ids[subtask_id] = False
        # 2. Parse energies
        try:
            mol.energies[subtask_id] = log.get_scf_energies(relative=False)[-1].item()
        except Exception as exc:
            # newer version may not able to parse scf energies
            # As a temporarily fix, set the energy to np.nan
            mol.energies[subtask_id] = np.nan
            print(f'Error in parsing energy of subtask {subtask_id} of {self.label}: {exc}')
        mol.frequencies[subtask_id] = log.freqs
