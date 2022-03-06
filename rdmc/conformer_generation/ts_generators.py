#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for ts conformer generation workflows
"""

import os
import numpy as np
import logging
from.utils import *
from .generators import StochasticConformerGenerator
from .pruners import TorsionPruner


class TSConformerGenerator:
    """
    stuff
    """
    def __init__(self, rxn_smiles, embedder=None, optimizer=None, pruner=None, save_dir=None):
        """

        Args:
            rxn_smiles:
            embedder:
            optimizer:
            pruner:
            save_dir:
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.rxn_smiles = rxn_smiles
        self.embedder = embedder
        self.optimizer = optimizer
        self.pruner = pruner
        self.save_dir = save_dir

    def embed_stable_species(self, smiles):

        smiles_list = smiles.split(".")

        mols = []
        for smi in smiles_list:
            n_conformers_per_iter = 20
            scg = StochasticConformerGenerator(
                smiles=smi,
                config="loose",
                pruner=TorsionPruner(),
                min_iters=5,
                max_iters=10,
            )

            confs = scg(n_conformers_per_iter)
            mol = dict_to_mol(confs)
            mols.append(mol)

        if len(mols) > 1:
            new_mol = [mols[0].CombineMol(m, offset=1, c_product=True) for m in mols[1:]][0]
            new_order = np.argsort(new_mol.GetAtomMapNumbers()).tolist()
            new_mol = new_mol.RenumberAtoms(new_order)
        else:
            new_mol = mols[0]

        return new_mol

    def __call__(self, n_conformers):

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        r_smi, p_smi = self.rxn_smiles.split(">>")

        self.logger.info("Embedding stable species conformers...")
        r_mol = self.embed_stable_species(r_smi)
        p_mol = self.embed_stable_species(p_smi)

        self.logger.info("Generating initial TS guesses...")
        ts_mol = self.embedder((r_mol, p_mol), n_conformers=n_conformers, save_dir=self.save_dir)

        self.logger.info("Optimizing TS guesses...")
        opt_ts_mol = self.optimizer(ts_mol, save_dir=self.save_dir)

        return opt_ts_mol
