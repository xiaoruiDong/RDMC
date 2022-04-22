#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for ts conformer generation workflows
"""

import os
import numpy as np
import logging
import random
from typing import List, Optional, Union

from rdmc.conformer_generation.utils import *
from rdmc.conformer_generation.generators import StochasticConformerGenerator
from rdmc.conformer_generation.pruners import *
from rdmc.conformer_generation.align import prepare_mols


class TSConformerGenerator:
    """
    The class used to define a workflow for generating a set of TS conformers.
    """

    def __init__(self,
                 rxn_smiles: str,
                 multiplicity: Optional[int] = None,
                 embedder: Optional['TSInitialGuesser'] = None,
                 optimizer: Optional['TSOptimizer'] = None,
                 pruner: Optional['ConfGenPruner'] = None,
                 verifiers: Optional[Union['TSVerifier',List['TSVerifier']]] = None,
                 save_dir: Optional[str] = None,
                 ) -> 'TSConformerGenerator':
        """
        Initiate the TS conformer generator object. The best practice is set all information here

        Args:
            rxn_smiles (str): The SMILES of the reaction. The SMILES should be formatted similar to `"reactant1.reactant2>>product1.product2."`.
            multiplicity (int, optional): The spin multiplicity of the reaction. The spin multiplicity will be interpreted from the reaction smiles if this
                                          is not given by the user.
            embedder (TSInitialGuesser, optional): The embedder used to generate TS initial guessers. Available options are `TSEGNNGuesser`, `TSGCNGuesser`.
                                                   `RMSDPPGuesser`, and `AutoNEBGuesser`.
            optimizer (TSOptimizer, optional): The optimizer used to optimize TS geometries. Available options are `SellaOptimizer`, `OrcaOptimizer`, and
                                               `GaussianOptimizer`.
            pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization. Available options are
                                              `CRESTPruner` and `TorsionPruner`.
            verifiers (TSVerifier or list of TSVerifiers, optional): The verifier or a list of verifiers used to verify the obtained TS conformer. Available
                                                                     options are `GaussianIRCVerifier`, `OrcaIRCVerifier`, and `XTBFrequencyVerifier`.
            save_dir (str or Pathlike object, optional): The path to save the intermediate files and outputs generated during the generation.
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.rxn_smiles = rxn_smiles
        if multiplicity:
           self.multiplicity = multiplicity
        else: 
            r_smi, p_smi = rxn_smiles.split(">>")
            r_mol = RDKitMol.FromSmiles(r_smi)
            p_mol = RDKitMol.FromSmiles(p_smi)
            r_mul = r_mol.GetSpinMultiplicity()
            p_mul = p_mol.GetSpinMultiplicity()
            if r_mul != p_mul:
                self.logger.warning(f"Inconsistent multiplicity!!")
                self.logger.warning(f"Reactants had multiplicty {r_mul}")
                self.logger.warning(f"Products had multiplicty {p_mul}")
                # use the smaller multiplicity
                mul = r_mul if r_mul < p_mul else p_mul
                logging.warning(f"Using multiplicity {mul} for all species...")
            self.multiplicity = mul
        self.embedder = embedder
        self.optimizer = optimizer
        self.pruner = pruner
        if isinstance(self.pruner, TorsionPruner):
            self.pruner.initialize_ts_torsions_list(rxn_smiles)
        elif isinstance(self.pruner, CRESTPruner):
            raise NotImplementedError("The CRESTPruner is not yes compatible with TS conformer generation!")

        self.verifiers = [] if not verifiers else verifiers
        self.save_dir = save_dir

    def embed_stable_species(self,
                             smiles: str,
                             ) -> 'rdmc.RDKitMol':
        """
        Embed the reactant and product complex according to the SMILES provided.

        Args:
            smiles (str): The reactant or product complex in SMILES. if multiple molecules involve,
                          use `.` to separate them.

        Returns:
            An RDKitMol of the reactant or product complex with 3D geometry embedded.
        """
        # Split the complex smiles into a list of molecule smiles
        smiles_list = smiles.split(".")

        # Create molecules
        mols = []
        for smi in smiles_list:

            rdmc_mol = RDKitMol.FromSmiles(smi)
            if len(rdmc_mol.GetTorsionalModes(includeRings=True)) < 1:
                pruner = CRESTPruner()
                min_iters = 1
            else:
                pruner = TorsionPruner()
                min_iters = 5

            # Since the reactant and the product is not the key here,
            # the ETKDG generator and only a loose threshold is used.
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
            # If more than one molecule is involved, combining them
            # their geometry back into one piece. c_product is set to true
            # so all combination are generated.
            new_mol = [mols[0].CombineMol(m, offset=1, c_product=True) for m in mols[1:]][0]
            # TODO: new_mol = new_mol.RenumberAtoms() can be replaced the following two lines
            # Make the change afterwards.
            new_order = np.argsort(new_mol.GetAtomMapNumbers()).tolist()
            new_mol = new_mol.RenumberAtoms(new_order)
        else:
            new_mol = mols[0]

        return new_mol

    def generate_seed_mols(self,
                           rxn_smiles: str,
                           n_conformers: int = 20,
                           ) -> list:
        """
        Genereate seeds of reactant/product pairs to be passed to the following steps.

        Args:
            rxn_smiles (str): The reaction smiles of the reaction.
            n_conformers (int, optional): The maximum number of conformers to be generated. Defaults to 20.

        Returns:
            list
        """
        # Convert SMILES to reactant and product complexes
        r_smi, p_smi = rxn_smiles.split(">>")
        r_mol = RDKitMol.FromSmiles(r_smi)
        p_mol = RDKitMol.FromSmiles(p_smi)

        # Generate information about number of reacants/products and number of rings
        n_reactants = len(r_mol.GetMolFrags())
        n_products = len(p_mol.GetMolFrags())
        n_reactant_rings = len([tuple(x) for x in r_mol.GetSymmSSSR()])
        n_product_rings = len([tuple(x) for x in p_mol.GetSymmSSSR()])

        # generate stable conformers depending on type of reaction
        # if more reactant fragments than product fragments (ex. A + B -> C), generate only product conformers
        if n_reactants > n_products:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        # if more product fragments than reactant fragments (ex. A -> B + C), generate only reactant conformers
        elif n_reactants < n_products:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        # if a ring forming/breaking reaction occurs, we want to start from the ring structure when generating TS
        # if more reactant has more rings, generate only reactant conformers
        elif n_reactant_rings > n_product_rings:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        # if more product has more rings, generate only product conformers
        elif n_reactant_rings < n_product_rings:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        else:
            n_reactant_confs = n_conformers // 2
            n_product_confs = n_conformers // 2

        # Create reactant conformer and product in pairs and store them as seed_mols
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

    def __call__(self,
                 n_conformers: int = 20):
        """
        Run the workflow of TS conformer generation.

        Args:
            n_conformers (int): The maximum number of conformers to be generated. Defaults to 20.
        """

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.logger.info("Embedding stable species conformers...")
        seed_mols = self.generate_seed_mols(self.rxn_smiles, n_conformers)

        # TODO: Need to double check if multiplicity is generally needed for embedder
        # It is needed for QST2, probably 
        self.logger.info("Generating initial TS guesses...")
        ts_mol = self.embedder(seed_mols, save_dir=self.save_dir)

        self.logger.info("Optimizing TS guesses...")
        opt_ts_mol = self.optimizer(ts_mol, multiplicity=self.multiplicity, save_dir=self.save_dir, rxn_smiles=self.rxn_smiles)

        if self.pruner:
            self.logger.info("Pruning TS guesses...")
            opt_ts_mol_dict, unique_ids = self.pruner(mol_to_dict(opt_ts_mol, energies=True), sort_by_energy=False, return_ids=True)
            keep_ids = [i in unique_ids for i in range(opt_ts_mol.GetNumConformers())]
            self.logger.info(f"Pruned {np.sum([x==False for x in keep_ids])} TS conformers")
            opt_ts_mol = dict_to_mol(opt_ts_mol_dict)
            self.logger.info(f'Keeping {len(keep_ids)} unique IDs...')

        self.logger.info("Verifying TS guesses...")
        if not self.pruner: keep_ids = [True] * opt_ts_mol.GetNumConformers()
        for verifier in self.verifiers:
            keep_ids = verifier(opt_ts_mol, keep_ids=keep_ids, multiplicity=self.multiplicity, save_dir=self.save_dir, rxn_smiles=self.rxn_smiles)

        return opt_ts_mol
