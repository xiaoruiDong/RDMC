#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
from rdmc.conformer_generation.verifier.base import BaseVerifier


class FreqVerifier(BaseVerifier):
    """
    The base frequency verifier. Useful if:
    - freqs are calculated in a previous task (e.g., optimization)
    - creating a more specific new task involving frequency calculation

    Args:
        cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
    """

    label = "FreqVerify"

    def task_prep(self,
                  cutoff_freq: float = -100.,
                  **kwargs,
                  ):
        """
        Set the method and cutoff.

        Args:

            cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
        """
        self.cutoff_freq = cutoff_freq

    def check_proceed(self,
                      mol: 'RDKitMol',
                      ) -> bool:
        """
        Check if the frequency verification should proceed.

        Args:
            mol ('RDKitMol') The molecule.

        Returns:
            bool: Whether the frequency verification should proceed.
        """
        # Uni-atom molecules don't have frequencies
        if mol.GetNumAtoms() == 1:
            return False

        # Check if there is any conformer succeeded in previous jobs
        # If keep id is not available, assume all succeeded
        if any([keep_id for keep_id in getattr(mol, 'keep_ids', [True])]):
            return True
        return False

    def check_freqs_avail(self,
                          mol: 'RDKitMol',
                          ) -> bool:
        """
        Check if the frequencies are available.

        Args:
            mol ('RDKitMol'): The molecule.

        Returns:
            bool: Whether the frequencies are available.
        """
        if not hasattr(mol, 'frequencies'):
            mol.frequencies = [None] * mol.GetNumConformers()
        if any([freqs is not None for freqs in mol.frequencies]):
            return True
        return False

    def check_negative_freqs(self,
                             freqs: 'np.ndarray',
                             ts: bool,
                             ) -> bool:
        """
        Check if the number of negative frequencies is correct.
        0 for non-TSs, and 1 for TSs.

        Args:
            freqs: The frequencies.
        Returns:
            bool: Whether there are negative frequencies.
        """
        if ts:
            return sum(freqs < self.cutoff_freq) == 1
        else:
            return not np.any(freqs < 0)

    def calc_freq(self,
                  mol: 'RDKitMol',
                  **kwargs):
        """
        Calculate the frequencies. This need to be implemented in the child class.
        The function at least need to pass the mol object as an argument.
        """
        raise NotImplementedError

    @BaseVerifier.timer
    def run(self,
            mol: 'RDKitMol',
            ts: bool = False,
            **kwargs):
        """
        Run the task.
        """
        # Check if there is a need to proceed
        # E.g.,
        # 1. Uni-atom molecules don't have frequencies
        # 2. If no conformers are kept, there is no need to proceed
        if not self.check_proceed(mol):
            return mol

        # Check if frequencies are available
        # If not, run the frequency calculation
        # that is implemented in the child class
        if not self.check_freqs_avail(mol):
            self.calc_freq(mol, **kwargs)

        # Check the frequencies one by one
        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            freqs = mol.frequencies[cid]
            mol.keep_ids[cid] = self.check_negative_freqs(freqs=freqs,
                                                          ts=ts)

        return mol

    @staticmethod
    def _get_mult_and_chrg(mol: 'RDKitMol',
                           multiplicity: Optional[int],
                           charge: Optional[int]):
        """
        Use the multiplicity and charge from the molecule if not specified
        """
        if multiplicity is None:
            multiplicity = mol.GetSpinMultiplicity()
        if charge is None:
            charge = mol.GetFormalCharge()
        return multiplicity, charge
