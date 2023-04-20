#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from typing import Optional

from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.conformer_generation.utils import timer, _software_available
from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
from rdmc.external.xtb_tools.utils import XTB_BINARY

_software_available['xtb'] = osp.isfile(XTB_BINARY)


class XTBOptimizer(BaseOptimizer):

    request_external_software = ['xtb']

    """
    Optimize conformers using the xTB software.
    """
    def __init__(self,
                 method: str = "gfn2",
                 level: str = "normal",
                 **kwargs):
        """
        Args:
            method (str): xTB method. Options: gfn0, gfn1, gfn2, gfnff. Defaults to gfn2,
                          which is equivalent to "--gfn 2" method when using xtb binary.
            level (str): optimization threshold. Options: crude, sloppy, loose, lax,
                         normal, tight, vtight, extreme. Defaults to normal.
        """
        super().__init__(method=method,
                         level=level,
                         **kwargs)

    def task_prep(self,
                  method: str = "gfn2",
                  level: str = "normal",
                  **kwargs,):
        """
        Prepare the task.
        """
        self.method = method
        self.level = level

    @timer
    def run(self,
            mol: 'RDKitMol',
            multiplicity: Optional[int] = None,
            charge: Optional[int] = None,
            **kwargs):
        """
        Optimize conformers using the xTB software.
        """
        # Correct multiplicity by user input
        # There are chances that the multiplicity in mol is not correct
        # multiplicity argument adds extra flexibility in calculation
        if multiplicity is None:
            uhf = mol.GetSpinMultiplicity() - 1
        else:
            uhf = multiplicity - 1
        if charge is None:
            charge = mol.GetFormalCharge()

        keep_ids = getattr(self, 'keep_ids', [True] * mol.GetNumConformers())

        new_mol = mol.Copy(quickCopy=True)
        self.energies = []
        for cid in range(self.n_subtasks):
            if not keep_ids[cid]:
                continue
            try:
                opt_mol, props = run_xtb_calc(mol,
                                              conf_id=cid,
                                              job="--opt",
                                              method=self.method,
                                              level=self.level,
                                              uhf=uhf,
                                              charge=charge,
                                              )
            except Exception as exc:
                keep_ids[cid] = False
                print(exc)
                raise
                # continue

            # Renumber the molecule based on the atom mapping just set
            conformer = opt_mol.GetConformer(id=0)
            conformer.SetIntProp('n_opt_cycles', props['n_opt_cycles'])
            new_mol.AddConformer(conformer._conf,
                                 assignId=True)

            energy = float(opt_mol.GetProp('total energy / Eh'))
            self.energies.append(energy)

        self.keep_ids = keep_ids

        return new_mol
