#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from typing import Optional

import numpy as np

from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.conformer_generation.utils import _software_available
from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
from rdmc.external.xtb_tools.utils import XTB_BINARY

_software_available['xtb'] = osp.isfile(XTB_BINARY)


class OldXTBOptimizer(BaseOptimizer):
    """
    Optimize conformers using the xTB software.

    Args:
    method (str): xTB method. Options: gfn0, gfn1, gfn2, gfnff. Defaults to gfn2,
                    which is equivalent to "--gfn 2" method when using xtb binary.
    level (str): optimization threshold. Options: crude, sloppy, loose, lax,
                    normal, tight, vtight, extreme. Defaults to normal.
    """

    request_external_software = ['xtb']

    def task_prep(self,
                  method: str = "gfn2",
                  level: str = "normal",
                  **kwargs,):
        """
        Prepare the task.
        """
        self.method = method
        self.level = level

    @BaseOptimizer.timer
    def run(self,
            mol: 'RDKitMol',
            multiplicity: Optional[int] = None,
            charge: Optional[int] = None,
            **kwargs):
        """
        Optimize conformers using the xTB software.
        """
        mult, charge = self._get_mult_and_chrg(mol, multiplicity, charge)

        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())

        new_mol = mol.Copy(copy_attrs=['keep_ids'])
        new_mol.energies = []
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                new_mol.energies.append(np.nan)
                continue
            try:
                opt_mol, props = run_xtb_calc(mol,
                                              conf_id=cid,
                                              job="--opt",
                                              method=self.method,
                                              level=self.level,
                                              uhf=mult - 1,
                                              charge=charge,
                                              )
            except Exception as exc:
                new_mol.energies.append(np.nan)
                new_mol.keep_ids[cid] = False
                print(exc)
                continue

            # Renumber the molecule based on the atom mapping just set
            new_mol.SetPositions(opt_mol.GetPositions(), id=cid)
            new_mol.GetConformer(cid).SetIntProp('n_opt_cycles', props['n_opt_cycles'])

            energy = float(opt_mol.GetProp('total energy / Eh'))
            new_mol.energies.append(energy)

        return new_mol
