#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import subprocess
import tempfile
from typing import Optional

import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.conformer_generation.utils import get_binary
from rdmc.external.logparser import ORCALog
from rdmc.external.inpwriter import write_orca_opt

orca_binary = get_binary('orca')


class ORCAOptimizer(BaseOptimizer):
    """
    The class to optimize geometries using the algorithm built in ORCA.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method available in ORCA to be used for TS optimization.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory. Defaults to XTB2.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): The memory to use in GB. Defaults to 1.
    """

    request_external_software = ['orca']

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  nprocs: int = 1,
                  memory: int = 1,
                  **kwargs,
                  ):
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

    @BaseOptimizer.timer
    def run(self,
            mol: 'RDKitMol',
            ts: bool = False,
            multiplicity: Optional[int] = None,
            charge: Optional[int] = None,
            **kwargs):
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            ts (bool, optional): Whether the molecule is a TS. Defaults to False.
            multiplicity (int): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
            charge (int): The charge of the molecule. Defaults to None, to use the charge of mol.

        Returns:
            RDKitMol
        """
        mult, charge = self._get_mult_and_chrg(mol, multiplicity, charge)

        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())
        work_dir = osp.abspath(self.save_dir) if self.save_dir else tempfile.mkdtemp()
        subtask_dirs = [osp.join(work_dir, f"orca_opt{cid}") for cid in range(len(run_ids))]
        input_paths = [osp.join(subtask_dirs[cid], "orca_opt.inp") for cid in range(len(run_ids))]
        log_paths = [osp.join(subtask_dirs[cid], "orca_opt.log") for cid in range(len(run_ids))]
        opt_xyzs = [osp.join(subtask_dirs[cid], "orca_opt.xyz") for cid in range(len(run_ids))]

        new_mol = mol.Copy(copy_attrs=[])
        new_mol.keep_ids = [False] * len(run_ids)
        new_mol.energies = [np.nan] * len(run_ids)
        new_mol.frequencies = [None] * len(run_ids)

        # Generate and save the ORCA input file
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            try:
                os.makedirs(subtask_dirs[cid], exist_ok=True)
                orca_input = write_orca_opt(mol=mol,
                                            conf_id=cid,
                                            ts=ts,
                                            charge=charge,
                                            mult=mult,
                                            method=self.method,
                                            memory=self.memory,
                                            **kwargs,
                                            )
                with open(input_paths[cid], 'w') as f_inp:
                    f_inp.write(orca_input)
            except Exception as exc:
                print(f'ORCA opt input file generation failed:\n{exc}')

        # Separate for loop
        # Todo: parallelize this
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue

            # Run the optimization using subprocess
            try:
                with open(log_paths[cid], "w") as f_log:
                    orca_run = subprocess.run(
                        args=[orca_binary, input_paths[cid]],
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        cwd=subtask_dirs[cid],
                        check=True,
                    )
            except Exception as exc:
                print(f'ORCA optimization failed:\n{exc}')
                continue

            # Check the Orca results
            try:
                orca_log = ORCALog(log_paths[cid])
                if orca_log.success:
                    opt_mol = RDKitMol.FromFile(opt_xyzs[cid], sanitize=ts)
                    new_mol.SetPositions(opt_mol.GetPositions(), id=cid)
                    new_mol.keep_ids[cid] = True
                    # If the energy and frequency are not available
                    # or the parsing failed, # the keep_ids and geometries
                    # will be still correctly set, respectively.
                    new_mol.energies[cid] = \
                                orca_log.get_scf_energies(converged=True,
                                                          relative=False)[-1].item()
                    new_mol.frequencies[cid] = orca_log.freqs
                else:
                    print('ORCA optimization succeeded but log parsing failed.')
            except Exception as exc:
                print(f'ORCA optimization succeeded but log parsing failed:\n{exc}')

        return new_mol
