#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for ts conformer generation workflows
"""

import os
import numpy as np
import logging
import random
from.utils import *
from .generators import StochasticConformerGenerator
from .pruners import *
from.align import prepare_mols


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

            rdmc_mol = RDKitMol.FromSmiles(smi)
            if len(rdmc_mol.GetTorsionalModes(includeRings=True)) < 1:
                pruner = CRESTPruner()
                min_iters = 1
            else:
                pruner = TorsionPruner()
                min_iters = 5

            n_conformers_per_iter = 20
            scg = StochasticConformerGenerator(
                smiles=smi,
                config="loose",
                pruner=pruner,
                min_iters=min_iters,
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

    def generate_seed_mols(self, rxn_smiles, n_conformers=20):

        r_smi, p_smi = rxn_smiles.split(">>")
        r_mol = RDKitMol.FromSmiles(r_smi)
        p_mol = RDKitMol.FromSmiles(p_smi)

        n_reactants = len(r_mol.GetMolFrags())
        n_products = len(p_mol.GetMolFrags())
        n_reactant_rings = len([tuple(x) for x in r_mol.GetSymmSSSR()])
        n_product_rings = len([tuple(x) for x in p_mol.GetSymmSSSR()])

        if n_reactants > n_products:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        elif n_reactants < n_products:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        elif n_reactant_rings > n_product_rings:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        elif n_reactant_rings < n_product_rings:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        else:
            n_reactant_confs = n_conformers // 2
            n_product_confs = n_conformers // 2

        seed_mols = []
        if n_reactant_confs > 0:
            r_embedded_mol = self.embed_stable_species(r_smi)
            r_embedded_mols = [r_embedded_mol.GetConformer(i).ToMol() for i in range(r_embedded_mol.GetNumConformers())]
            random.shuffle(r_embedded_mols)
            rp_combos = [prepare_mols(r, RDKitMol.FromSmiles(p_smi)) for r in r_embedded_mols[:n_reactant_confs]]
            seed_mols.extend(rp_combos)

        if n_product_confs > 0:
            p_embedded_mol = self.embed_stable_species(p_smi)
            p_embedded_mols = [p_embedded_mol.GetConformer(i).ToMol() for i in range(p_embedded_mol.GetNumConformers())]
            random.shuffle(p_embedded_mols)
            pr_combos = [prepare_mols(p, RDKitMol.FromSmiles(r_smi)) for p in p_embedded_mols[:n_product_confs]]
            seed_mols.extend(pr_combos)

        return seed_mols

    def __call__(self, n_conformers):

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.logger.info("Embedding stable species conformers...")
        seed_mols = self.generate_seed_mols(self.rxn_smiles, n_conformers)

        self.logger.info("Generating initial TS guesses...")
        ts_mol = self.embedder(seed_mols, save_dir=self.save_dir)

        self.logger.info("Optimizing TS guesses...")
        opt_ts_mol = self.optimizer(ts_mol, save_dir=self.save_dir)

        return opt_ts_mol
