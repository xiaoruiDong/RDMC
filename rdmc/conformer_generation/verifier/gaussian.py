#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import shutil
import subprocess
import tempfile
from typing import Optional

from rdmc.conformer_generation.verifier.freq import FreqVerifier
from rdmc.external.inpwriter import write_gaussian_freq
from rdmc.conformer_generation.utils import get_binary, _software_available
from rdmc.external.logparser import GaussianLog

gaussian_binaries = {binname: get_binary(binname) for binname in ['g16', 'g09', 'g03']}


class GaussianFreqVerifier(FreqVerifier):
    """
    The class for verifying the species or TS by calculating and checking its frequencies using Gaussian.
    Since frequency may be calculated in an previous job. The class will first check if frequency
    results are available. If not, it will launch jobs to calculate frequencies.

    Args:
        method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
        cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                     imaginary frequency to avoid small magnitude frequencies which correspond
                                     to internal bond rotations. This is only used when the molecule is a TS.
                                     Defaults to -100 cm^-1.
    """

    # Note, request_external_software is not used for
    # check_external_software as other tasks
    request_external_software = ['g16', 'g09', 'g03']

    def task_prep(self,
                  method: str = "gfn2-xtb",
                  nprocs: int = 1,
                  memory: int = 1,
                  cutoff_freq: float = -100.,
                  **kwargs,
                  ):
        """
        Set the method, number of processors, memory and cutoff.

        Args:
            method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
            nprocs (int, optional): The number of processors used to calculate frequencies. Defaults to 1.
            memory (int, optional): The memory used to calculate frequencies in GB. Defaults to 1.
            cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
        """
        super().task_prep(cutoff_freq=cutoff_freq)
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

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

    def calc_freq(self,
                  mol: 'RDKitMol',
                  multiplicity: Optional[int] = None,
                  charge: Optional[int] = None,
                  **kwargs):
        """
        Calculate the frequencies using Gaussian.

        Args:
            mol (RDKitMol)
            multiplicity (int): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
            charge (int): The charge of the molecule. Defaults to None, to use the charge of mol.
        """
        mult, charge = self._get_mult_and_chrg(mol, multiplicity, charge)

        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())
        work_dir = osp.abspath(self.save_dir) if self.save_dir else tempfile.mkdtemp()
        subtask_dirs = [osp.join(work_dir, f"gaussian_freq{cid}") for cid in range(len(run_ids))]
        input_paths = [osp.join(subtask_dirs[cid], "gaussian_freq.gjf") for cid in range(len(run_ids))]
        log_paths = [osp.join(subtask_dirs[cid], "gaussian_freq.log") for cid in range(len(run_ids))]

        # Generate and save the gaussian input files
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            try:
                os.makedirs(subtask_dirs[cid], exist_ok=True)
                gaussian_input = write_gaussian_freq(mol=mol,
                                                     conf_id=cid,
                                                     charge=charge,
                                                     mult=mult,
                                                     method=self.method,
                                                     memory=self.memory,
                                                     **kwargs,)
                with open(input_paths[cid], 'w') as f_inp:
                    f_inp.write(gaussian_input)
            except Exception as exc:
                mol.keep_ids[cid] = False
                print(f'Gaussian freq input file generation failed:\n{exc}')

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
                print(f'Gaussian freq calculation failed:\n{exc}')
                continue

            # Check the gaussian optimization results
            try:
                glog = GaussianLog(log_paths[cid])
                if glog.success:
                    mol.frequencies[cid] = glog.freqs
                else:
                    mol.keep_ids[cid] = False
                    print('Gaussian optimization succeeded but log parsing failed.')
            except Exception as exc:
                mol.keep_ids[cid] = False
                print(f'Gaussian optimization succeeded but log parsing failed:\n{exc}')

        # Clean up
        if not self.save_dir:
            shutil.rmtree(work_dir)
