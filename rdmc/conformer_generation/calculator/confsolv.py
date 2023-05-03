#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for including solvation corrections
"""

import os.path as osp
import torch

from rdmc.conformer_generation.task import MolTask
from rdmc.conformer_generation.utils import register_software

with register_software("conf_solv"):
    from conf_solv.trainer import LitConfSolvModule
    from conf_solv.dataloaders.collate import Collater
    from conf_solv.dataloaders.loader import create_pairdata, MolGraph

kJ_to_kcal = 0.239006


class ConfSolvCalculator(MolTask):
    """
    Class for estimating conformer energies in solution with neural networks.
    """

    init_attrs = {'solv_energies': 0.0}

    def task_prep(self,
                  model_dir: str,
                  **kwargs):
        """
        Load the conf_solve model.

        Args:
            model_dir (str): The path to the directory storing the trained ConfSolv model.
        """
        self.model_dir = model_dir
        # Load the model and custom collater
        self.module = LitConfSolvModule.load_from_checkpoint(
            checkpoint_path=osp.join(model_dir, "best_model.ckpt"),
        )
        self.module.model.eval()
        self.collater = Collater(follow_batch=["x_solvent", "x_solute"],
                                 exclude_keys=None)

    @MolTask.timer
    def run(self,
            mol: 'RDKitMol',
            solvent_smi: str = 'O',
            **kwargs,
            ) -> 'RDKitMol':
        """
        Run the SolvConf model to estimate the solvation energies of the conformers.
        This method will update the solv_energies attribute of the mol.

        Args:
            mol (RDKitMol): The molecule to be estimated.
            solvent_smi (str): The SMILES string of the solvent. Defaults to 'O', for water.

        Returns:
            mol (RDKitMol): The updated molecule with solvation energies.
        """
        # Generate representations of the solvent and solute
        solute_confs = [mol.ToAtoms(cid) for cid in self.run_ids]
        solvent_mol = MolGraph(solvent_smi)
        pair_data = create_pairdata(solvent_mol,
                                    solute_confs,
                                    self.n_subtasks)
        data = self.collater([pair_data])

        # Run the model and obtain the solvation energies
        with torch.no_grad():
            abs_energies = self.module(data, self.n_subtasks).numpy() * kJ_to_kcal
        rel_energies = abs_energies - abs_energies.min()

        # update mol with the solvation energies
        for eid, cid in enumerate(self.run_ids):
            mol.solv_energies[cid] = rel_energies[eid]

        return mol
