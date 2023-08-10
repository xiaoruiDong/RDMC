#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for verifying optimized stable species
"""

# Import RDMC first to avoid unexpected errors
from rdmc import RDKitMol

import os
import pickle
from glob import glob
import subprocess
from time import time
from typing import Optional

from rdmc.external.xtb_tools.opt import run_xtb_calc


class Verifier:
    """
    The abstract class for verifiers.
    """
    def __init__(self,
                 track_stats: bool = False):
        """
        Initialize the verifier.

        Args:
            track_stats (bool, optional): Whether to track status. Defaults to False.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_guesses(self,
                       mol: 'RDKitMol',
                       multiplicity: int = 1,
                       save_dir: Optional[str] = None,
                       **kwargs):
        """
        The abstract method for verifying guesses (or optimized stable species geometries). The method need to take
        `mol` in RDKitMol, `keep_ids` in list, `multiplicity` in int, and `save_dir` in str, and returns
        a list indicating the ones passing the check.

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def __call__(self,
                 mol: 'RDKitMol',
                 multiplicity: int = 1,
                 save_dir: Optional[str] = None,
                 **kwargs):
        """
        Run the workflow for verifying the stable species guessers (or optimized stable species conformers).

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            list: a list of true and false
        """
        time_start = time()
        mol = self.verify_guesses(
                mol=mol,
                multiplicity=multiplicity,
                save_dir=save_dir,
                **kwargs
            )

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return mol


class XTBFrequencyVerifier(Verifier):
    """
    The class for verifying the stable species by calculating and checking its frequencies using XTB.
    """
    def __init__(self,
                 cutoff_frequency: int = -100,
                 track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (int, optional): Cutoff frequency above which a frequency does not correspond to a TS
                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                (defaults to -100 cm-1)
            track_stats (bool, optional): Whether to track stats. Defaults to False.
        """
        super(XTBFrequencyVerifier, self).__init__(track_stats)

        self.cutoff_frequency = cutoff_frequency

    def verify_guesses(self,
                       mol: 'RDKitMol',
                       multiplicity: int = 1,
                       save_dir: Optional[str] = None,
                       **kwargs):
        """
        Verifying stable species guesses (or optimized stable species geometries).

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            RDKitMol
        """
        if mol.GetNumAtoms() != 1:
            for i in range(mol.GetNumConformers()):
                if mol.KeepIDs[i]:
                    if mol.frequency[i] is None:
                        props = run_xtb_calc(mol, confId=i, job="--hess", uhf=multiplicity - 1)
                        frequencies = props["frequencies"]
                    else:
                        frequencies = mol.frequency[i]

                    freq_check = sum(frequencies < self.cutoff_frequency) == 0
                    mol.KeepIDs[i] = freq_check

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(mol.KeepIDs, f)

        return mol
