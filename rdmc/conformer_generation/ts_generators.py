#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for TS conformer generation workflows.
"""

import logging
import os
import pickle
from typing import List, Optional, Union
import random

import numpy as np

from rdmc import RDKitMol

from rdmc.conformer_generation.utils import *
from rdmc.conformer_generation.generators import StochasticConformerGenerator
from rdmc.conformer_generation.pruners import *
from rdmc.conformer_generation.align import prepare_mols


class TSConformerGenerator:
    """
    The class used to define a workflow for generating a set of TS conformers.

    Args:
        rxn_smiles (str): The SMILES of the reaction. The SMILES should be formatted similar to ``"reactant1.reactant2>>product1.product2."``.
        multiplicity (int, optional): The spin multiplicity of the reaction. The spin multiplicity will be interpreted from the reaction smiles if this
                                      is not given by the user.
        use_smaller_multiplicity (bool, optional): Whether to use the smaller multiplicity when the interpreted multiplicity from the reaction smiles is
                                                   inconsistent between reactants and products. Defaults to ``True``.
        embedder (TSInitialGuesser, optional): Instance of a :obj:`TSInitialGuesser <rdmc.conformer_generation.ts_guessers.TSInitialGuesser>`. Available options are
                                               :obj:`TSEGNNGuesser <rdmc.conformer_generation.ts_guessers.TSEGNNGuesser>`,
                                               :obj:`TSGCNGuesser <rdmc.conformer_generation.ts_guessers.TSGCNGuesser>`,
                                               :obj:`AutoNEBGuesser <rdmc.conformer_generation.ts_guessers.AutoNEBGuesser>`,
                                               :obj:`RMSDPPGuesser <rdmc.conformer_generation.ts_guessers.RMSDPPGuesser>`, and
                                               :obj:`DEGSMGuesser <rdmc.conformer_generation.ts_guessers.DEGSMGuesser>`.
        optimizer (TSOptimizer, optional): Instance of a :obj:`TSOptimizer <rdmc.conformer_generation.ts_optimizers.TSOptimizer>`. Available options are
                                           :obj:`SellaOptimizer <rdmc.conformer_generation.ts_optimizers.SellaOptimizer>`,
                                           :obj:`OrcaOptimizer <rdmc.conformer_generation.ts_optimizers.OrcaOptimizer>`,
                                           :obj:`GaussianOptimizer <rdmc.conformer_generation.ts_optimizers.GaussianOptimizer>`, and
                                           :obj:`QChemOptimizer <rdmc.conformer_generation.ts_optimizers.QChemOptimizer>`.
        pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization.
                                          Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                          :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                          :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
        verifiers (TSVerifier or list of TSVerifiers, optional): The verifier or a list of verifiers used to verify the obtained TS conformer.
                                                                 Instance of a :obj:`TSVerifier <rdmc.conformer_generation.ts_verifiers.TSVerifier>`.
                                                                 Available options are
                                                                 :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.ts_verifiers.XTBFrequencyVerifier>`,
                                                                 :obj:`GaussianIRCVerifier <rdmc.conformer_generation.ts_verifiers.GaussianIRCVerifier>`,
                                                                 :obj:`OrcaIRCVerifier <rdmc.conformer_generation.ts_verifiers.OrcaIRCVerifier>`,
                                                                 :obj:`QChemIRCVerifier <rdmc.conformer_generation.ts_verifiers.QChemIRCVerifier>`, and
                                                                 :obj:`TSScreener <rdmc.conformer_generation.ts_verifiers.TSScreener>`.
        sampler (TorisonalSampler, optional): The sampler used to do automated conformer search for the obtained TS conformer. You can use
                                              :obj:`TorsionalSampler <rdmc.conformer_generation.torsional_sampling.TorsionalSampler>` to define your own sampler.
        final_modules (TSOptimizer, TSVerifier or list of TSVerifiers, optional): The final modules can include optimizer in different LoT than previous
                                                                                  one and verifier(s) used to verify the obtained TS conformer.
        save_dir (str or Pathlike object, optional): The path to save the intermediate files and outputs generated during the generation. Defaults to ``None``.
    """

    def __init__(self,
                 rxn_smiles: str,
                 multiplicity: Optional[int] = None,
                 use_smaller_multiplicity: bool = True,
                 embedder: Optional['TSInitialGuesser'] = None,
                 optimizer: Optional['TSOptimizer'] = None,
                 pruner: Optional['ConfGenPruner'] = None,
                 verifiers: Optional[Union['TSVerifier', List['TSVerifier']]] = None,
                 sampler: Optional['TorisonalSampler'] = None,
                 final_modules: Optional[Union['TSOptimizer', 'TSVerifier', List['TSVerifier']]] = None,
                 save_dir: Optional[str] = None,
                 ) -> 'TSConformerGenerator':
        """
        The class used to define a workflow for generating a set of TS conformers.

        Args:
            rxn_smiles (str): The SMILES of the reaction. The SMILES should be formatted similar to ``"reactant1.reactant2>>product1.product2."``.
            multiplicity (int, optional): The spin multiplicity of the reaction. The spin multiplicity will be interpreted from the reaction smiles if this
                                        is not given by the user.
            use_smaller_multiplicity (bool, optional): Whether to use the smaller multiplicity when the interpreted multiplicity from the reaction smiles is
                                                       inconsistent between reactants and products. Defaults to ``True``.
            embedder (TSInitialGuesser, optional): Instance of a :obj:`TSInitialGuesser <rdmc.conformer_generation.ts_guessers.TSInitialGuesser>`. Available options are
                                                   :obj:`TSEGNNGuesser <rdmc.conformer_generation.ts_guessers.TSEGNNGuesser>`,
                                                   :obj:`TSGCNGuesser <rdmc.conformer_generation.ts_guessers.TSGCNGuesser>`,
                                                   :obj:`AutoNEBGuesser <rdmc.conformer_generation.ts_guessers.AutoNEBGuesser>`,
                                                   :obj:`RMSDPPGuesser <rdmc.conformer_generation.ts_guessers.RMSDPPGuesser>`, and
                                                   :obj:`DEGSMGuesser <rdmc.conformer_generation.ts_guessers.DEGSMGuesser>`.
            optimizer (TSOptimizer, optional): Instance of a :obj:`TSOptimizer <rdmc.conformer_generation.ts_optimizers.TSOptimizer>`. Available options are
                                               :obj:`SellaOptimizer <rdmc.conformer_generation.ts_optimizers.SellaOptimizer>`,
                                               :obj:`OrcaOptimizer <rdmc.conformer_generation.ts_optimizers.OrcaOptimizer>`,
                                               :obj:`GaussianOptimizer <rdmc.conformer_generation.ts_optimizers.GaussianOptimizer>`, and
                                               :obj:`QChemOptimizer <rdmc.conformer_generation.ts_optimizers.QChemOptimizer>`.
            pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization.
                                              Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                              :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                              :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
            verifiers (TSVerifier or list of TSVerifiers, optional): The verifier or a list of verifiers used to verify the obtained TS conformer.
                                                                     Instance of a :obj:`TSVerifier <rdmc.conformer_generation.ts_verifiers.TSVerifier>`.
                                                                     Available options are
                                                                     :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.ts_verifiers.XTBFrequencyVerifier>`,
                                                                     :obj:`GaussianIRCVerifier <rdmc.conformer_generation.ts_verifiers.GaussianIRCVerifier>`,
                                                                     :obj:`OrcaIRCVerifier <rdmc.conformer_generation.ts_verifiers.OrcaIRCVerifier>`,
                                                                     :obj:`QChemIRCVerifier <rdmc.conformer_generation.ts_verifiers.QChemIRCVerifier>`, and
                                                                     :obj:`TSScreener <rdmc.conformer_generation.ts_verifiers.TSScreener>`.
            sampler (TorisonalSampler, optional): The sampler used to do automated conformer search for the obtained TS conformer. You can use
                                                  :obj:`TorsionalSampler <rdmc.conformer_generation.torsional.TorsionalSampler>` to define your own sampler.
            final_modules (TSOptimizer, TSVerifier or list of TSVerifiers, optional): The final modules can include optimizer in different LoT than previous
                                                                                      one and verifier(s) used to verify the obtained TS conformer.
            save_dir (str or Pathlike object, optional): The path to save the intermediate files and outputs generated during the generation. Defaults to ``None``.
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
            mul = r_mul
            if r_mul != p_mul:
                self.logger.warning(f"Inconsistent multiplicity!!")
                self.logger.warning(f"Reactants had multiplicty {r_mul}")
                self.logger.warning(f"Products had multiplicty {p_mul}")
                if use_smaller_multiplicity:
                    # use the smaller multiplicity
                    mul = r_mul if r_mul < p_mul else p_mul
                else:
                    # use the larger multiplicity
                    mul = r_mul if r_mul > p_mul else p_mul
                logging.warning(f"Using multiplicity {mul} for all species...")
            self.multiplicity = mul
        self.embedder = embedder
        self.optimizer = optimizer
        self.pruner = pruner
        if isinstance(self.pruner, TorsionPruner):
            self.pruner.initialize_ts_torsions_list(rxn_smiles)

        self.verifiers = [] if not verifiers else verifiers
        self.sampler = sampler
        self.final_modules = [] if not final_modules else final_modules
        self.save_dir = save_dir

    def embed_stable_species(self,
                             smiles: str,
                             ) -> 'RDKitMol':
        """
        Embed the reactant and product complex according to the SMILES provided.

        Args:
            smiles (str): The reactant or product complex in SMILES. if multiple molecules involve,
                          use ``"."`` to separate them.

        Returns:
            RDKitMol: An RDKitMol of the reactant or product complex with 3D geometry embedded.
        """
        # Split the complex smiles into a list of molecule smiles
        smiles_list = smiles.split(".")

        # Create molecules
        mols = []
        for smi in smiles_list:

            rdmc_mol = RDKitMol.FromSmiles(smi)

            # use default embedder if only one atom present
            if rdmc_mol.GetNumAtoms() == 1:
                mol = rdmc_mol.Copy(quickCopy=True)
                mol.EmbedConformer()
                mols.append(mol)
                continue

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
                           shuffle: bool = False,
                           ) -> list:
        """
        Genereate seeds of reactant/product pairs to be passed to the following steps.

        Args:
            rxn_smiles (str): The reaction smiles of the reaction.
            n_conformers (int, optional): The maximum number of conformers to be generated. Defaults to ``20``.
            shuffle (bool, optional): Whether or not to shuffle the embedded mols. Defaults to ``False``.

        Returns:
            list: A list of reactant/product pairs in ``RDKitMol``.
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
        # if a ring forming/breaking reaction occurs, we want to start from the ring structure when generating TS
        # if more reactant has more rings, generate only reactant conformers
        if n_reactant_rings > n_product_rings:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        # if more product has more rings, generate only product conformers
        elif n_reactant_rings < n_product_rings:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        # if more reactant fragments than product fragments (ex. A + B -> C), generate only product conformers
        elif n_reactants > n_products:
            n_reactant_confs = 0
            n_product_confs = n_conformers

        # if more product fragments than reactant fragments (ex. A -> B + C), generate only reactant conformers
        elif n_reactants < n_products:
            n_reactant_confs = n_conformers
            n_product_confs = 0

        else:
            n_reactant_confs = n_conformers // 2
            n_product_confs = n_conformers // 2

        # Create reactant conformer and product in pairs and store them as seed_mols
        seed_mols = []
        if n_reactant_confs > 0:
            r_embedded_mol = self.embed_stable_species(r_smi)
            r_embedded_mols = [r_embedded_mol.GetEditableConformer(i).ToMol() for i in range(r_embedded_mol.GetNumConformers())]
            random.shuffle(r_embedded_mols) if shuffle else None
            rp_combos = [prepare_mols(r, RDKitMol.FromSmiles(p_smi)) for r in r_embedded_mols[:n_reactant_confs]]
            [(r.SetProp("Identity", "reactant"), p.SetProp("Identity", "product"))
                for (r, p) in rp_combos]  # to properly save r/p to file later
            seed_mols.extend(rp_combos)

        if n_product_confs > 0:
            p_embedded_mol = self.embed_stable_species(p_smi)
            p_embedded_mols = [p_embedded_mol.GetEditableConformer(i).ToMol() for i in range(p_embedded_mol.GetNumConformers())]
            random.shuffle(p_embedded_mols) if shuffle else None
            pr_combos = [prepare_mols(p, RDKitMol.FromSmiles(r_smi)) for p in p_embedded_mols[:n_product_confs]]
            [(p.SetProp("Identity", "product"), r.SetProp("Identity", "reactant"))
                for (p, r) in pr_combos]  # to properly save r/p to file later
            seed_mols.extend(pr_combos)

        return seed_mols

    def set_filter(self,
                   ts_mol: 'RDKitMol',
                   n_conformers: int,
                   ) -> RDKitMol:
        """
        Assign the indices of reactions to track whether the conformers are passed to the following steps.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            n_conformers (int): The maximum number of conformers to be passed to the following steps.

        Returns:
            RDKitMol: with ``KeepIDs`` as a list of ``True`` and ``False`` indicating whether a conformer passes the check.
        """
        energy_dict = ts_mol.energy
        KeepIDs = ts_mol.KeepIDs

        sorted_index = [k for k, v in sorted(energy_dict.items(), key=lambda item: item[1])]  # Order by energy
        filter_index = [k for k in sorted_index if KeepIDs[k]][:n_conformers]
        for i in range(ts_mol.GetNumConformers()):
            if i not in filter_index:
                ts_mol.KeepIDs[i] = False
        return ts_mol

    def __call__(self,
                 n_conformers: int = 20,
                 n_verifies: int = 20,
                 n_sampling: int = 1,
                 n_refines: int = 1,
                 ) -> 'RDKitMol':
        """
        Run the workflow of TS conformer generation.

        Args:
            n_conformers (int): The maximum number of conformers to be generated. Defaults to ``20``.
            n_verifies (int): The maximum number of conformers to be passed to the verifiers.  Defaults to ``20``.
            n_sampling (int): The maximum number of conformers to be passed to the torsional sampling. Defaults to ``1``.
            n_refines (int): The maximum number of conformers to be passed to the final modules. Defaults to ``1``.

        Returns:
            RDKitMol: The TS in RDKitMol object with 3D geometries embedded.
        """

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.logger.info("Embedding stable species conformers...")
        seed_mols = self.generate_seed_mols(self.rxn_smiles, n_conformers)

        # TODO: Need to double check if multiplicity is generally needed for embedder
        # It is needed for QST2, probably
        self.logger.info("Generating initial TS guesses...")
        ts_mol = self.embedder(seed_mols, multiplicity=self.multiplicity, save_dir=self.save_dir)
        ts_mol.KeepIDs = {i: True for i in range(ts_mol.GetNumConformers())}  # map ids of generated guesses thru workflow

        self.logger.info("Optimizing TS guesses...")
        opt_ts_mol = self.optimizer(ts_mol, multiplicity=self.multiplicity, save_dir=self.save_dir, rxn_smiles=self.rxn_smiles)

        if self.pruner:
            self.logger.info("Pruning TS guesses...")
            _, unique_ids = self.pruner(mol_to_dict(opt_ts_mol, conf_copy_attrs=["KeepIDs", "energy"]),
                                        sort_by_energy=False, return_ids=True)
            self.logger.info(f"Pruned {self.pruner.n_pruned_confs} TS conformers")
            opt_ts_mol.KeepIDs = {k: k in unique_ids and v for k, v in opt_ts_mol.KeepIDs.items()}
            with open(os.path.join(self.save_dir, "prune_check_ids.pkl"), "wb") as f:
                pickle.dump(opt_ts_mol.KeepIDs, f)

        self.logger.info("Verifying TS guesses...")
        opt_ts_mol = self.set_filter(opt_ts_mol, n_verifies)
        for verifier in self.verifiers:
            verifier(opt_ts_mol, multiplicity=self.multiplicity, save_dir=self.save_dir, rxn_smiles=self.rxn_smiles)

        # run torsional sampling
        if self.sampler:
            self.logger.info("Running torsional sampling...")
            energy_dict = opt_ts_mol.energy
            KeepIDs = opt_ts_mol.KeepIDs
            sorted_index = [k for k, v in sorted(energy_dict.items(), key=lambda item: item[1])]  # Order by energy
            filter_index = [k for k in sorted_index if KeepIDs[k]][:n_sampling]
            found_lower_energy_index = {i: False for i in range(opt_ts_mol.GetNumConformers())}
            for id in filter_index:
                original_energy = opt_ts_mol.energy[id]
                new_mol = self.sampler(opt_ts_mol, id, rxn_smiles=self.rxn_smiles, save_dir=self.save_dir)
                new_energy = new_mol.energy[id]
                if new_energy < original_energy:
                    found_lower_energy_index[id] = True
            # save which ts found conformer with lower energy via torsional sampling
            with open(os.path.join(self.save_dir, "sampler_check_ids.pkl"), "wb") as f:
                pickle.dump(found_lower_energy_index, f)

            # Doesn't find any lower energy conformer! Using original result...
            if not all(value is False for value in found_lower_energy_index.values()):
                opt_ts_mol = new_mol

        self.logger.info("Running final modules...")
        if self.final_modules:
            opt_ts_mol = self.set_filter(opt_ts_mol, n_refines)
            save_dir = os.path.join(self.save_dir, "final_modules")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for module in self.final_modules:
                opt_ts_mol = module(opt_ts_mol, multiplicity=self.multiplicity, save_dir=save_dir, rxn_smiles=self.rxn_smiles)

        # save which ts passed full workflow
        with open(os.path.join(self.save_dir, "workflow_check_ids.pkl"), "wb") as f:
            pickle.dump(opt_ts_mol.KeepIDs, f)

        return opt_ts_mol
