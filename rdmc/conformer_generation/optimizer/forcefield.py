#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.forcefield import RDKitFF


class MMFFOptimizer(BaseOptimizer):
    """
    Optimize conformers using the MMFF force field.

    Args:
        software (str): software to use for optimization. Options: rdkit, openbabel.
        variant (str): MMFF variant. Options: MMFF94, MMFF94s.
    """
    def task_prep(self,
                  software: str = "rdkit",
                  variant: str = "MMFF94s",
                  **kwargs):
        """
        Prepare the task.
        """
        self.software = software
        if software == 'rdkit':
            self.ff = RDKitFF(force_field=variant)
        elif software == 'openbabel':
            raise NotImplementedError

    @BaseOptimizer.timer
    def run(self,
            mol: 'RDKitMol',
            **kwargs):
        """
        Optimize conformers using the MMFF force field.

        Args:
            mol (RDKitMol): RDKitMol object.
        """
        opt_mol = mol.Copy()
        self.ff.setup(opt_mol)
        results = self.ff.optimize_confs(**kwargs)
        opt_mol = self.ff.get_optimized_mol()
        keep_ids, opt_mol.energies = zip(*results)  # kcal/mol
        opt_mol.keep_ids = [s == 0 for s in keep_ids]

        return opt_mol
