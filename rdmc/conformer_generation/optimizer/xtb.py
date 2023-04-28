#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.task import XTBBaseTask


class XTBOptimizer(XTBBaseTask, IOOptimizer):
    """
    Optimize conformers using the xTB software.

    Args:
    method (str): xTB method. Options: gfn0, gfn1, gfn2, gfnff. Defaults to gfn2,
                    which is equivalent to "--gfn 2" method when using xtb binary.
    level (str): optimization threshold. Options: crude, sloppy, loose, lax,
                    normal, tight, vtight, extreme. Defaults to normal.
    """

    label = 'XTBOptimizer'
    subtask_dir_name = 'xtb_opt'
    create_mol_flag = True
    init_attrs = {'energies': np.nan,}
    keep_files = ['xtbout.json', 'xtbout.log', 'xtbopt.log', 'g98.out']
    calc_type = '--opt'

    def save_data(self, **kwargs):
        """
        Save the data.
        """
        super(XTBBaseTask, self).save_data(**kwargs)  # from IOOptimizer

    def post_run(self, **kwargs):
        """
        Setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super(XTBBaseTask, self).post_run(**kwargs)  # from IOOptimizer

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the xTB output file and set them to the molecule.
        """
        opt_mol, props = subtask_result
        mol.SetPositions(opt_mol.GetPositions(), id=subtask_id)
        mol.GetConformer(subtask_id).SetIntProp('n_opt_cycles', props['n_opt_cycles'])
        energy = float(opt_mol.GetProp('total energy / Eh'))
        mol.energies[subtask_id] = energy
