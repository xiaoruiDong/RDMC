#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for including solvation corrections
"""

import os
from typing import Optional
from ase import Atoms
from time import time
import torch

try:
    from conf_solv.trainer import LitConfSolvModule
    from conf_solv.dataloaders.collate import Collater
    from conf_solv.dataloaders.loader import create_pairdata, MolGraph
except:
    print("No ConfSolv installation detected. Skipping import...")


class Estimator:
    """
    The abstract class for energy estimator.
    """
    def __init__(self,
                 track_stats: Optional[bool] = False):
        """
        Initialize the TS optimizer.
        """
        self.track_stats = track_stats
        self.stats = []

    def predict_energies(self,
                         mol_data: dict,
                         **kwargs):
        """
        The abstract method for predicting energies. It will be implemented in actual classes.
        The method needs to take `mol_data` which is a dictionary containing info about the
        conformers of the moelcule. It will return the molecule as the same 'mol_data' object
        with the energy values altered.

        Args:
            mol_data (list): A list of molecule dictionaries.

        Returns:
            mol_data
        """
        raise NotImplementedError

    def __call__(self,
                 mol_data: dict,
                 **kwargs):
        """
        Run the workflow to predict energies.

        Args:
            mol_data (list): A list of molecule dictionaries.

        Returns:
            mol_data
        """
        time_start = time()
        updated_mol_data = self.predict_energies(mol_data=mol_data, **kwargs)

        # sort by energy
        updated_mol_data = sorted(updated_mol_data, key=lambda x: x["energy"])

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return updated_mol_data


class ConfSolv(Estimator):
    """
    Class for estimating conformer energies in solution with neural networks.
    """

    def __init__(self,
                 trained_model_dir: str,
                 track_stats: Optional[bool] = False):
        """
        Initialize the ConfSolv model.

        Args:
            trained_model_dir (str): The path to the directory storing the trained ConfSolv model.
            track_stats (bool, optional): Whether to track timing stats. Defaults to False.
        """
        super(ConfSolv, self).__init__(track_stats)

        # Load the model and custom collater
        self.module = LitConfSolvModule.load_from_checkpoint(
            checkpoint_path=os.path.join(trained_model_dir, "best_model.ckpt"),
        )
        self.module.model.eval()
        self.collater = Collater(follow_batch=["x_solvent", "x_solute"], exclude_keys=None)

    def predict_energies(self,
                         mol_data: list,
                         **kwargs):
        """
        Predict conformer free energies in given solvent.

        Args:
            mol_data (list): A list of molecule dictionaries.

        Returns:
            mol_data
        """
        # prepare inputs
        syms = [a.GetSymbol() for a in mol_data[0]['conf'].ToMol().GetAtoms()]
        positions = [x['positions'] for x in mol_data]
        mols = [Atoms(symbols=syms, positions=pos) for pos in positions]
        solvent_molgraph = MolGraph(kwargs['solvent_smi'])
        pair_data = create_pairdata(solvent_molgraph, mols, len(mols))
        data = self.collater([pair_data])

        # make predictions of relative free energies
        with torch.no_grad():
            abs_energies = self.module(data, len(mols)).numpy() * 0.239006  # kcal/mol
        rel_energies = abs_energies - abs_energies.min()

        # update mol data
        [mol_data[i].update({'energy': rel_energies[i]}) for i in range(len(mols))]

        return mol_data
