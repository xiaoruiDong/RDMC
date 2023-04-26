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
from rdmc.conformer_generation.utils import get_binary, _software_available
from rdmc.external.logparser import GaussianLog
from rdmc.external.inpwriter import write_gaussian_opt


gaussian_binaries = {binname: get_binary(binname) for binname in ['g16', 'g09', 'g03']}


class GaussianOptimizer(BaseOptimizer):
    """
    The class to optimize geometries using the algorithm built in Gaussian.
    You have to have the Gaussian package installed to run this optimizer.
    # todo: make a general optimizer for Gaussian, QChem, and ORCA

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
        gaussian_binary (str, optional): The name of the gaussian binary, useful when there is multiple versions of Gaussian installed.
                                         Defaults to the latest version found in the environment variables.
    """

    request_external_software = ['g16', 'g09', 'g03']  # Not used for check_external_software that is specially written for Gaussian

    def check_external_software(self,
                                **kwargs,):
        """
        Check if Gaussian is installed.
        If the user specifies a Gaussian binary, use it.
        """
        # If the user specifies a Gaussian binary
        user_req_bin = _software_available.get(kwargs.get('gaussian_binary'))
        if user_req_bin:
            self.gaussian_binary = user_req_bin
        # Use the latest Gaussian binary found in the environment variables
        for binname in ['g16', 'g09', 'g03']:
            if _software_available.get(binname):
                self.gaussian_binaries = gaussian_binaries[binname]
                break
        else:
            raise RuntimeError('No Gaussian installation found.')

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  nprocs: int = 1,
                  memory: int = 1,):
        """
        Set up the Gaussian optimizer.
        """
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
        Optimize the geometries or TS guesses.

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
        subtask_dirs = [osp.join(work_dir, f"gaussian_opt{cid}") for cid in range(len(run_ids))]
        input_paths = [osp.join(subtask_dirs[cid], "gaussian_opt.gjf") for cid in range(len(run_ids))]
        log_paths = [osp.join(subtask_dirs[cid], "gaussian_opt.log") for cid in range(len(run_ids))]

        new_mol = mol.Copy(copy_attrs=[])
        new_mol.keep_ids = [False] * len(run_ids)
        new_mol.energies = [np.nan] * len(run_ids)
        new_mol.frequencies = [None] * len(run_ids)

        # Generate and save the gaussian input files
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            try:
                os.makedirs(subtask_dirs[cid], exist_ok=True)
                gaussian_input = write_gaussian_opt(mol=mol,
                                                    conf_id=cid,
                                                    ts=ts,
                                                    charge=charge,
                                                    mult=mult,
                                                    method=self.method,
                                                    memory=self.memory,
                                                    **kwargs,
                                                    )
                with open(input_paths[cid], 'w') as f_inp:
                    f_inp.write(gaussian_input)
            except Exception as exc:
                print(f'Gaussin opt input file generation failed:\n{exc}')

        # Separate for loop for future parallelization
        # Todo: parallelize this
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue

            # Run the optimization using subprocess
            try:
                with open(log_paths[cid], "w") as f_log:
                    gaussian_run = subprocess.run(
                        args=[self.gaussian_binary, input_paths[cid]],
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        cwd=subtask_dirs[cid],
                        check=True,
                    )
            except Exception as exc:
                print(f'Gaussian optimization failed:\n{exc}')
                continue

            # Check the gaussian optimization results
            try:
                glog = GaussianLog(log_paths[cid])
                if glog.success:
                    opt_mol = glog.get_mol(embed_conformers=False,
                                           converged=True,
                                           sanitize=False)
                    new_mol.SetPositions(opt_mol.GetPositions(), id=cid)
                    new_mol.keep_ids[cid] = True
                    new_mol.energies[cid] = \
                                glog.get_scf_energies(converged=True,
                                                      relative=False)[-1].item()
                    new_mol.frequencies[cid] = glog.freqs
                else:
                    print('Gaussian optimization succeeded but log parsing failed.')
            except Exception as exc:
                print(f'Gaussian optimization succeeded but log parsing failed:\n{exc}')

        # Clean up
        if not self.save_dir:
            shutil.rmtree(work_dir)

        return new_mol
