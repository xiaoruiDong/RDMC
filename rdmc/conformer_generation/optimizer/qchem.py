#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.optimizer.base import BaseOptimizer
from rdmc.conformer_generation.utils import get_binary
from rdmc.external.logparser import QChemLog
from rdmc.external.inpwriter import write_qchem_opt

qchem_binary = get_binary('qchem')


class QChemOptimizer(BaseOptimizer):
    """
    The class to optimize geometries using the algorithm built in QChem.
    You have to have the QChem package installed and run `source qcenv.sh` to run this optimizer.
    # todo: make a general optimizer for Gaussian, QChem, and ORCA

    Args:
        method (str, optional): The method available in ORCA to be used for TS optimization.
                                Defaults to wb97x-d3.
        basis (str, optional): The basis set to use. Defaults to def2-svp.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
    """

    request_external_software = ['qchem']

    def task_prep(self,
                  method: str = "wb97x-d3",
                  basis: str = 'def2-svp',
                  nprocs: int = 1,
                  **kwargs,
                  ):
        self.method = method
        self.basis = basis
        self.nprocs = nprocs

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
        subtask_dirs = [osp.join(work_dir, f"qchem_opt{cid}") for cid in range(len(run_ids))]
        input_paths = [osp.join(subtask_dirs[cid], "qchem_opt.qcin") for cid in range(len(run_ids))]
        log_paths = [osp.join(subtask_dirs[cid], "qchem_opt.log") for cid in range(len(run_ids))]

        new_mol = mol.Copy(copy_attrs=[])
        new_mol.keep_ids = [False] * len(run_ids)
        new_mol.energies = [np.nan] * len(run_ids)
        new_mol.frequencies = [None] * len(run_ids)

        # Generate and save the Qchem input file
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            try:
                os.makedirs(subtask_dirs[cid], exist_ok=True)
                qchem_input = write_qchem_opt(mol=mol,
                                              conf_id=cid,
                                              ts=ts,
                                              charge=charge,
                                              mult=mult,
                                              method=self.method,
                                              basis=self.basis,
                                              **kwargs,
                                              )
                with open(input_paths[cid], 'w') as f_inp:
                    f_inp.write(qchem_input)
            except Exception as exc:
                print(f'QChem opt input file generation failed:\n{exc}')

        # Separate for loop
        # Todo: parallelize this
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue

            # Run the optimization using subprocess
            try:
                with open(log_paths[cid], "w") as f_log:
                    qchem_run = subprocess.run(
                        args=[qchem_binary, "-nt", str(self.nprocs), input_paths[cid]],
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        cwd=subtask_dirs[cid],
                        check=True,
                    )
            except Exception as exc:
                print(f'QChem optimization failed:\n{exc}')
                continue

            # Check the QChem results
            try:
                qchem_log = QChemLog(log_paths[cid])
                if qchem_log.success:
                    opt_mol = qchem_log.get_mol(embed_conformers=False,
                                                sanitize=False)
                    new_mol.SetPositions(opt_mol.GetPositions(), id=cid)
                    new_mol.keep_ids[cid] = True
                    new_mol.energies[cid] = \
                                qchem_log.get_scf_energies(relative=False)[-1].item()
                    new_mol.frequencies[cid] = qchem_log.freqs
                else:
                    print('QChem optimization succeeded but log parsing failed.')
            except Exception as exc:
                print(f'QChem optimization succeeded but log parsing failed:\n{exc}')

        # Clean up
        if not self.save_dir:
            shutil.rmtree(work_dir)

        return new_mol
