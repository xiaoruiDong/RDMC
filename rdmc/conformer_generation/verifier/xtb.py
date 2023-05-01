#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from typing import Optional

from rdmc.conformer_generation.task import XTBBaseTask
from rdmc.conformer_generation.verifier.freq import FreqVerifier


class XTBFreqVerifier(FreqVerifier, XTBBaseTask):
    """
    The class for verifying the species or TS by calculating and checking its frequencies using XTB.
    Since frequency may be calculated in an previous job. The class will first check if frequency
    results are available. If not, it will launch jobs to calculate frequencies.

    Args:
        method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
        cutoff_freq (float, optional): A cutoff frequency determine whether a imaginary frequency
                                       is a valid mode. Only used for TS verification. Defaults to -100 cm^-1,
                                       that is imaginary frequencies between -100 to 0 cm^-1 are
                                       considered not valid reaction mode.
    """

    subtask_dir = 'xtb_freq'

    def task_prep(self,
                  **kwargs,
                  ):
        """
        Set the method and cutoff.

        Args:
            method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
            cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
        """
        super().task_prep(**kwargs)  # FreqVerifier.task_prep
        super(FreqVerifier, self).task_prep(**kwargs)  # XTBBaseTask.task_prep

    def runner(self,
               mol: 'RDKitMol',
               subtask_id: int,
               **kwargs,
               ) -> tuple:
        """
        The runner of each subtask using `run_xtb_calc` in RDMC.external.xtb_tools.

        For developers: In the xTB freq verifier implementation,
        It will first check if the frequencies are already calculated. If not,
        it will call `run_xtb_calc` (XTBBaseTask's runner) defined in
        RDMC.external.xtb_tools to calculate frequencies. Also as a note,
        kwargs will always have keys of charge and mult, as they are
        assigned by the second step of `run` in `MolIOTask`. Be careful if you need to
        reimplement `run` or `runner`.

        Args:
            mol ('RDKitMol') The molecule.
            subtask_id (int) The id of the subtask.
            kwargs: Other arguments.

        Returns:
            tuple: The molecule and its properties.
        """
        if self.need_calc_freqs(mol=mol, subtask_id=subtask_id):
            # Not using subprocess runner but XTBBaseTask.runner
            return super(FreqVerifier, self).runner(mol=mol, subtask_id=subtask_id, **kwargs)

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Calculate the frequencies using xTB.

        Args:
            mol (RDKitMol)
            multiplicity (int): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
            charge (int): The charge of the molecule. Defaults to None, to use the charge of mol.
        """
        if self.need_calc_freqs(mol, subtask_id):
            mol.frequencies[subtask_id] = subtask_result[1].get("frequencies")
        mol.keep_ids[subtask_id] = self.check_negative_freqs(freqs=mol.frequencies[subtask_id],
                                                             ts=kwargs.get('ts') or False)
