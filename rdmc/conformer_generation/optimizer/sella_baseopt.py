#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import redirect_stdout
import io
import os.path as osp
import shutil
import tempfile

from ase.calculators.orca import ORCA
import numpy as np
import pandas as pd

from rdmc import RDKitMol
from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.conformer_generation.utils import register_software

with register_software('sella'):
    from sella import Sella

with register_software('xtb-python'):
    from xtb.ase.calculator import XTB
    from xtb.utils import get_method


class SellaOptimizer(BaseOptimizer):
    """
    The class to optimize TS geometries using the Sella algorithm.
    It uses XTB as the backend calculator, ASE as the interface, and Sella module from the Sella repo.

    Args:
        method (str, optional): The method in XTB used to optimize the geometry. Options are 'GFN1-xTB' and 'GFN2-xTB'. Defaults to "GFN2-xTB".
        fmax (float, optional): The force threshold used in the optimization. Defaults to 1e-3.
        steps (int, optional): Max number of steps allowed in the optimization. Defaults to 1000.
    """

    request_external_software = ['sella', 'xtb-python']

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  fmax: float = 1e-3,
                  steps: int = 1000,):
        """
        Set up the Sella optimizer.
        """
        self.method = method
        self.fmax = fmax
        self.steps = steps

    @BaseOptimizer.timer
    def run(self,
            mol: 'RDKitMol',
            **kwargs):
        """
        Optimize the TS guesses. You need to correctly set the unpaired electrons and formal charges
        for each atom in the mol.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.

        Returns:
            RDKitMol
        """
        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())
        work_dir = osp.abspath(self.save_dir) if self.save_dir else tempfile.mkdtemp()
        subtask_dirs = [osp.join(work_dir, f"sella_opt{cid}") for cid in range(len(run_ids))]
        traj_paths = [osp.join(subtask_dirs[cid], "sella_opt.traj") for cid in range(len(run_ids))]
        log_paths = [osp.join(subtask_dirs[cid], "sella_opt.log") for cid in range(len(run_ids))]
        orca_names = [osp.join(subtask_dirs[cid], "sella_opt") for cid in range(len(run_ids))]  # Xiaorui doesn't know the usage

        new_mol = mol.Copy(copy_attrs=[])
        new_mol.keep_ids = [False] * len(run_ids)
        new_mol.energies = [np.nan] * len(run_ids)
        new_mol.frequencies = [None] * len(run_ids)

        xtb_method = get_method(self.method)

        # Todo: parallelize this
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue

            atoms = mol.ToAtoms(confId=cid)
            atoms.set_initial_magnetic_moments(
                        [atom.GetNumRadicalElectrons() + 1
                         for atom in mol.GetAtoms()])
            atoms.set_initial_charges(
                        [atom.GetFormalCharge()
                         for atom in mol.GetAtoms()])

            if xtb_method:
                atoms.calc = XTB(method=xtb_method)
            else:
                atoms.calc = ORCA(label=orca_names[cid],
                                  orcasimpleinput=self.method)

            try:
                # Run the optimization using subprocess
                with io.StringIO() as buf, redirect_stdout(buf):
                    opt_atoms = Sella(atoms,
                                      logfile=log_paths[cid],
                                      trajectory=traj_paths[cid],
                                      )
                    opt_atoms.run(self.fmax, self.steps)
            except Exception as exc:
                print(f'Sella optimization failed:\n{exc}')
                continue

            # Read the optimized geometry and results
            try:
                new_mol.SetPositions(opt_atoms.atoms.positions, id=cid)
                new_mol.keep_ids[cid] = True
                energy = float(pd.read_csv(log_paths[cid]).iloc[-1].values[0].split()[3])  # Need an example
                new_mol.energies[cid] = energy
            except Exception as exc:
                print(f'Sella optimization finished but log parsing failed:\n{exc}')

        # Clean up
        if not self.save_dir:
            shutil.rmtree(work_dir)

        return new_mol
