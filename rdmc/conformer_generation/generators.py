#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for conformer generation workflows
"""
import logging
from time import time
from typing import Optional

import numpy as np

from rdmc.mol import RDKitMol
from .embedders import *
from .pruners import *
from .optimizers import *
from .metrics import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)


class StochasticConformerGenerator:
    """
    A module for stochastic conformer generation. The workflow follows an embed -> optimize -> prune cycle with
    custom stopping criteria. Additional final modules can be added at the user's discretion.

    Args:
        smiles (str): SMILES input for which to generate conformers.
        embedder (ConfGenEmbedder, optional): Instance of a :obj:`ConfGenEmbedder <rdmc.conformer_generation.embedders.ConfGenEmbedder>`.
                                              Available options are :obj:`ETKDGEmbedder <rdmc.conformer_generation.embedders.ETKDGEmbedder>`,
                                              :obj:`GeoMolEmbedder <rdmc.conformer_generation.embedders.GeoMolEmbedder>`, and
                                              :obj:`RandomEmbedder <rdmc.conformer_generation.embedders.RandomEmbedder>`.
        optimizer (ConfGenOptimizer, optional): Instance of a :obj:`ConfGenOptimizer <rdmc.conformer_generation.optimizers.ConfGenOptimizer>`.
                                                Available options are :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>`,
                                                :obj:`GaussianOptimizer <rdmc.conformer_generation.optimizers.GaussianOptimizer>`, and
                                                :obj:`MMFFOptimizer <rdmc.conformer_generation.optimizers.MMFFOptimizer>`.
        estimator (Estimator, optional): Instance of an :obj:`Estimator <rdmc.conformer_generation.solvation.Estimator>`. Available option is
                                         :obj:`ConfSolv <rdmc.conformer_generation.solvation.ConfSolv>`.
        pruner (ConfGenPruner, optional): Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                          :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                          :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
        metric (SCGMetric, optional): The available option is `SCGMetric <rdmc.conformer_generation.metrics.SCGMetric>`.
        min_iters (int, optional): Minimum number of iterations for which to run the module.
        max_iters (int, optional}: Maximum number of iterations for which to run the module.
        final_modules (list): List of instances of optimizer/pruner to run after initial cycles complete.
    """
    def __init__(self,
                 smiles,
                 embedder: Optional['ConfGenEmbedder'] = None,
                 optimizer: Optional['ConfGenOptimizer'] = None,
                 estimator: Optional['Estimator'] = None,
                 pruner: Optional['ConfGenPruner'] = None,
                 metric: Optional['SCGMetric'] = None,
                 min_iters: Optional[int] = None,
                 max_iters: Optional[int] = None,
                 final_modules: Optional[list] = None,
                 config: Optional[dict] = None,
                 track_stats: bool = False):
        """
        Initialize the StochasticConformerGenerator module.

        Args:
            smiles (str): SMILES input for which to generate conformers.
            embedder (ConfGenEmbedder, optional): Instance of a :obj:`ConfGenEmbedder <rdmc.conformer_generation.embedders.ConfGenEmbedder>`.
                                                Available options are :obj:`ETKDGEmbedder <rdmc.conformer_generation.embedders.ETKDGEmbedder>`,
                                                :obj:`GeoMolEmbedder <rdmc.conformer_generation.embedders.GeoMolEmbedder>`, and
                                                :obj:`RandomEmbedder <rdmc.conformer_generation.embedders.RandomEmbedder>`.
            optimizer (ConfGenOptimizer, optional): Instance of a :obj:`ConfGenOptimizer <rdmc.conformer_generation.optimizers.ConfGenOptimizer>`.
                                                    Available options are :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>`,
                                                    :obj:`GaussianOptimizer <rdmc.conformer_generation.optimizers.GaussianOptimizer>`, and
                                                    :obj:`MMFFOptimizer <rdmc.conformer_generation.optimizers.MMFFOptimizer>`.
            estimator (Estimator, optional): Instance of an :obj:`Estimator <rdmc.conformer_generation.solvation.Estimator>`. Available option is
                                            :obj:`ConfSolv <rdmc.conformer_generation.solvation.ConfSolv>`.
            pruner (ConfGenPruner, optional): Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                            :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                            :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
            metric (SCGMetric, optional): The available option is `SCGMetric <rdmc.conformer_generation.metrics.SCGMetric>`.
            min_iters (int, optional): Minimum number of iterations for which to run the module.
            max_iters (int, optional}: Maximum number of iterations for which to run the module.
            final_modules (list): List of instances of optimizer/pruner to run after initial cycles complete.
        """

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.smiles = smiles

        self.embedder = embedder
        self.optimizer = optimizer
        self.estimator = estimator
        self.pruner = pruner
        self.metric = metric
        self.final_modules = [] if not final_modules else final_modules
        self.track_stats = track_stats

        if config:
            self.logger.info(f"Config specified: using default settings for {config} config")
            self.set_config(config, embedder, optimizer, pruner, metric, final_modules, min_iters, max_iters)

        if not self.optimizer:
            self.metric.metric = "total conformers"
            self.logger.info("No optimizer selected: termination criteria set to total conformers")

        self.mol = RDKitMol.FromSmiles(smiles)
        self.unique_mol_data = []
        self.stats = []
        self.min_iters = 1 if not min_iters else min_iters
        self.max_iters = 1000 if not max_iters else max_iters
        self.iter = 0

        if isinstance(self.pruner, TorsionPruner):
            self.pruner.initialize_torsions_list(smiles)

    def __call__(self,
                 n_conformers_per_iter: int,
                 **kwargs,
                 ) -> List[dict]:
        """
        Run the workflow for stochastic conformer generation.

        Args:
            n_conformers_per_iter (int): The number of conformers to generate per iteration.

        Returns:
            unique_mol_data (List[dict]): A list of dictionaries containing the unique conformers.
        """
        self.logger.info(f"Generating conformers for {self.smiles}")
        time_start = time()
        for _ in range(self.max_iters):
            self.iter += 1

            self.logger.info(f"\nIteration {self.iter}: embedding {n_conformers_per_iter} initial guesses...")
            initial_mol_data = self.embedder(self.smiles, n_conformers_per_iter)

            if self.optimizer:
                self.logger.info(f"Iteration {self.iter}: optimizing initial guesses...")
                opt_mol_data = self.optimizer(initial_mol_data)
            else:
                opt_mol_data = []
                for c_id in range(len(initial_mol_data)):
                    conf = initial_mol_data[c_id]["conf"]
                    positions = conf.GetPositions()
                    opt_mol_data.append({"positions": positions,
                                         "conf": conf,
                                         "energy": np.nan})

            # check for failures
            if len(opt_mol_data) == 0:
                self.logger.info("Failed to optimize any of the embedded conformers")
                continue

            self.logger.info(f"Iteration {self.iter}: pruning conformers...")
            unique_mol_data = self.pruner(opt_mol_data, self.unique_mol_data) if self.pruner else opt_mol_data
            unique_mol_data = self.estimator(unique_mol_data, **kwargs) if self.estimator else unique_mol_data
            self.metric.calculate_metric(unique_mol_data)
            self.unique_mol_data = unique_mol_data
            self.logger.info(f"Iteration {self.iter}: kept {len(unique_mol_data)} unique conformers")

            if self.iter < self.min_iters:
                continue

            if self.metric.check_metric():
                self.logger.info(f"Iteration {self.iter}: stop crietria reached\n")
                break

        if self.iter == self.max_iters:
            self.logger.info(f"Iteration {self.iter}: max iterations reached\n")
        for module in self.final_modules:
            self.logger.info(f"Calling {module.__class__.__name__}")
            unique_mol_data = module(unique_mol_data)
            self.unique_mol_data = unique_mol_data

        if self.track_stats:
            time_end = time()
            stats = {"iter": self.iter,
                     "time": time_end - time_start,
                     }
            self.stats.append(stats)

        return unique_mol_data

    def set_config(self,
                   config: str,
                   embedder: Optional['ConfGenEmbedder'] = None,
                   optimizer: Optional['ConfGenOptimizer'] = None,
                   pruner: Optional['ConfGenPruner'] = None,
                   metric: Optional['SCGMetric'] = None,
                   min_iters: Optional[int] = None,
                   max_iters: Optional[int] = None,
                   final_modules: Optional[list] = None,
                   ):
        """
        Set the configuration for the conformer generator with pre-defined options: ``"loose"`` and ``"normal"``.

        Args:
            embedder (ConfGenEmbedder, optional): Instance of a :obj:`ConfGenEmbedder <rdmc.conformer_generation.embedders.ConfGenEmbedder>`.
                                                Available options are :obj:`ETKDGEmbedder <rdmc.conformer_generation.embedders.ETKDGEmbedder>`,
                                                :obj:`GeoMolEmbedder <rdmc.conformer_generation.embedders.GeoMolEmbedder>`, and
                                                :obj:`RandomEmbedder <rdmc.conformer_generation.embedders.RandomEmbedder>`.
                                                Defaults to :obj:`ETKDGEmbedder <rdmc.conformer_generation.embedders.ETKDGEmbedder>`.
            optimizer (ConfGenOptimizer, optional): Instance of a :obj:`ConfGenOptimizer <rdmc.conformer_generation.optimizers.ConfGenOptimizer>`.
                                                    Available options are :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>`,
                                                    :obj:`GaussianOptimizer <rdmc.conformer_generation.optimizers.GaussianOptimizer>`, and
                                                    :obj:`MMFFOptimizer <rdmc.conformer_generation.optimizers.MMFFOptimizer>`. Defaults to 
                                                    :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>` with ``"gff"`` method.
            pruner (ConfGenPruner, optional): Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                              :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                              :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`. By default,
                                              ``"loose"`` utilizes :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>` with 
                                              ``mean_chk_threshold=20`` and ``max_chk_threshold=30``, and ``"normal"`` utilizes
                                              :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>`.
            metric (SCGMetric, optional): The available option is `SCGMetric <rdmc.conformer_generation.metrics.SCGMetric>`.
                                          By default, both modes use ``"entropy"`` metric, while ``"loose"`` with ``window=3`` and ``threshold=0.05``,
                                          and ``"normal"`` with ``window=5`` and ``threshold=0.01``.
            min_iters (int, optional): Minimum number of iterations for which to run the module. Defaults to ``3`` for ``"loose"`` and ``5`` for ``"normal"``.
            max_iters (int, optional}: Maximum number of iterations for which to run the module. Defaults to ``20`` for ``"loose"`` and ``100`` for ``"normal"``.
            final_modules (list): List of instances of optimizer/pruner to run after initial cycles complete. By default, no final modules are used for ``"loose"``.
                                  For ``"normal"``, two :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` with ``ewin=12`` and ``ewin=6``, are introduced
                                  before and after a :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>` with ``"gfn2"`` method
                                  and ``"vtight"`` level.
        """
        if config == "loose":
            self.embedder = ETKDGEmbedder() if not embedder else embedder
            self.optimizer = XTBOptimizer(method="gff") if not optimizer else optimizer
            self.pruner = TorsionPruner(mean_chk_threshold=20, max_chk_threshold=30) if not pruner else pruner
            self.metric = SCGMetric(metric="entropy", window=3, threshold=0.05) if not metric else metric
            self.final_modules = [] if not final_modules else final_modules
            self.min_iters = 3 if not min_iters else min_iters
            self.max_iters = 20 if not max_iters else max_iters

        elif config == "normal":
            self.embedder = ETKDGEmbedder(track_stats=self.track_stats) if not embedder else embedder
            self.optimizer = XTBOptimizer(method="gff", track_stats=self.track_stats) if not optimizer else optimizer
            self.pruner = CRESTPruner(track_stats=self.track_stats)
            self.metric = SCGMetric(metric="entropy", window=5, threshold=0.01) if not metric else metric
            self.final_modules = [
                CRESTPruner(ewin=12, track_stats=self.track_stats),
                XTBOptimizer(method="gfn2", level="vtight", track_stats=self.track_stats),
                CRESTPruner(ewin=6, track_stats=self.track_stats)
            ] if not final_modules else final_modules
            self.min_iters = 5 if not min_iters else min_iters
            self.max_iters = 100 if not max_iters else max_iters

class ConformerGenerator:
    """
    A module for conformer generation. The workflow follows an embed -> optimize -> prune cycle with
    custom stopping criteria. Additional final modules can be added at the user's discretion.

    Args:
        smiles (str): SMILES input for which to generate conformers.
        multiplicity (int, optional): The spin multiplicity of the species. Defaults to ``None``,
                                      which will be interpreted from molecule generated by the `smiles`.
        optimizer (ConfGenOptimizer, optional): Instance of a :obj:`ConfGenOptimizer <rdmc.conformer_generation.optimizers.ConfGenOptimizer>`.
                                                Available options are :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>`,
                                                :obj:`GaussianOptimizer <rdmc.conformer_generation.optimizers.GaussianOptimizer>`, and
                                                :obj:`MMFFOptimizer <rdmc.conformer_generation.optimizers.MMFFOptimizer>`.
        pruner (ConfGenPruner, optional): Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                          :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                          :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
        verifiers (Verifier, optional): Instance of a :obj:`Verifier <rdmc.conformer_generation.verifiers.Verifier>`.
                                        Available option is :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.verifiers.XTBFrequencyVerifier>`.
        sampler (TorsionalSampler, optional): Instance of a :obj:`TorsionalSampler <rdmc.conformer_generation.sampler.TorsionalSampler>`.
        final_modules (list): List of instances of optimizer/pruner to run after initial cycles complete.
        save_dir (str or Pathlike object, optional): The path to save the intermediate files and outputs generated during the generation.
    """
    def __init__(self,
                 smiles: str,
                 multiplicity: Optional[int] = None,
                 optimizer: Optional['ConfGenOptimizer'] = None,
                 pruner: Optional['ConfGenPruner'] = None,
                 verifiers: Optional[Union['Verifier',List['Verifier']]] = None,
                 sampler: Optional['TorisonalSampler'] = None,
                 final_modules: Optional[Union['ConfGenOptimizer','Verifier']] = None,
                 save_dir: Optional[str] = None,
                 ) -> 'ConformerGenerator':
        """
        Initiate the conformer generator object. The best practice is set all information here.

        Args:
            smiles (str): SMILES input for which to generate conformers.
            multiplicity (int, optional): The spin multiplicity of the species. Defaults to ``None``,
                                        which will be interpreted from molecule generated by the `smiles`.
            optimizer (ConfGenOptimizer, optional): Instance of a :obj:`ConfGenOptimizer <rdmc.conformer_generation.optimizers.ConfGenOptimizer>`.
                                                    Available options are :obj:`XTBOptimizer <rdmc.conformer_generation.optimizers.XTBOptimizer>`,
                                                    :obj:`GaussianOptimizer <rdmc.conformer_generation.optimizers.GaussianOptimizer>`, and
                                                    :obj:`MMFFOptimizer <rdmc.conformer_generation.optimizers.MMFFOptimizer>`.
            pruner (ConfGenPruner, optional): Instance of a :obj:`ConfGenPruner <rdmc.conformer_generation.pruners.ConfGenPruner>`. Available options are
                                            :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                            :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
            verifiers (Verifier, optional): Instance of a :obj:`Verifier <rdmc.conformer_generation.verifiers.Verifier>`.
                                            Available option is :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.verifiers.XTBFrequencyVerifier>`.
            sampler (TorsionalSampler, optional): Instance of a :obj:`TorsionalSampler <rdmc.conformer_generation.sampler.TorsionalSampler>`.
            final_modules (list): List of instances of optimizer/pruner to run after initial cycles complete.
            save_dir (str or Pathlike object, optional): The path to save the intermediate files and outputs generated during the generation.
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.smiles = smiles
        if multiplicity:
           self.multiplicity = multiplicity
        else:
            mol = RDKitMol.FromSmiles(smiles)
            mul = mol.GetSpinMultiplicity()
            self.multiplicity = mul
        self.optimizer = optimizer
        self.pruner = pruner
        if isinstance(self.pruner, TorsionPruner):
            self.pruner.initialize_torsions_list(smiles)

        self.verifiers = [] if not verifiers else verifiers
        self.sampler = sampler
        self.final_modules = [] if not final_modules else final_modules
        self.save_dir = save_dir

    def embed_stable_species(self,
                             smiles: str,
                             n_conformers: int = 20,
                             ) -> 'RDKitMol':
        """
        Embed the well conformer according to the SMILES provided.

        Args:
            smiles (str): The well conformer SMILES.
            n_conformers (int, optional): The maximum number of conformers to be generated. Defaults to ``20``.

        Returns:
            An RDKitMol of the well conformer with 3D geometry embedded.
        """
        rdmc_mol = RDKitMol.FromSmiles(smiles)

        # use default embedder if only one atom present
        if rdmc_mol.GetNumAtoms() == 1:
            mol = rdmc_mol.Copy(quickCopy=True)
            mol.EmbedConformer()
            return mol

        if len(rdmc_mol.GetTorsionalModes(includeRings=True)) < 1:
            pruner = CRESTPruner()
            min_iters = 1
        else:
            pruner = TorsionPruner()
            min_iters = 5

        n_conformers_per_iter = 20
        scg = StochasticConformerGenerator(
            smiles=smiles,
            config="loose",
            pruner=pruner,
            min_iters=min_iters,
            max_iters=10,
        )

        confs = scg(n_conformers_per_iter)
        energies = [data['energy'] for data in confs]
        sort_index = np.argsort(energies)
        mol = dict_to_mol([confs[i] for i in sort_index[:n_conformers]])
        return mol

    def set_filter(self,
                   mol: RDKitMol,
                   n_conformers: int,
                   ) -> list:
        """
        Assign the indices of conformers to track whether the conformers are passed to the following steps.

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            n_conformers (int): The maximum number of conformers to be passed to the following steps.

        Returns:
            RDKitMol: with ``KeepIDs`` as a list of ``True`` and ``False`` indicating whether a conformer passes the check.
        """
        energy_dict = mol.energy
        KeepIDs = mol.KeepIDs

        sorted_index = [k for k, v in sorted(energy_dict.items(), key = lambda item: item[1])]  # Order by energy
        filter_index = [k for k in sorted_index if KeepIDs[k]][:n_conformers]
        for i in range(mol.GetNumConformers()):
            if i not in filter_index:
                mol.KeepIDs[i] = False
        return mol

    def __call__(self,
                 n_conformers: int = 20,
                 n_verifies: int = 20,
                 n_sampling: int = 1,
                 n_refines: int = 1):
        """
        Run the workflow of well conformer generation.

        Args:
            n_conformers (int): The maximum number of conformers to be generated. Defaults to ``20``.
            n_verifies (int): The maximum number of conformers to be passed to the verifiers. Defaults to ``20``.
            n_sampling (int): The maximum number of conformers to be passed to the torsional sampling. Defaults to ``1``.
            n_refines (int): The maximum number of conformers to be passed to the final modules. Defaults to ``1``.
        """

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.logger.info("Embedding stable species conformers...")
        mol = self.embed_stable_species(self.smiles, n_conformers)

        mol.KeepIDs = {i: True for i in range(mol.GetNumConformers())}  # map ids of generated guesses thru workflow

        self.logger.info("Optimizing stable species guesses...")
        opt_mol = self.optimizer(mol, multiplicity=self.multiplicity, save_dir=self.save_dir)

        if self.pruner:
            self.logger.info("Pruning stable species guesses...")
            _, unique_ids = self.pruner(mol_to_dict(opt_mol, conf_copy_attrs=["KeepIDs", "energy"]),
                                        sort_by_energy=False, return_ids=True)
            self.logger.info(f"Pruned {self.pruner.n_pruned_confs} well conformers")
            opt_mol.KeepIDs = {k: k in unique_ids and v for k, v in opt_mol.KeepIDs.items()}
            with open(os.path.join(self.save_dir, "prune_check_ids.pkl"), "wb") as f:
                pickle.dump(opt_mol.KeepIDs, f)

        self.logger.info("Verifying stable species guesses...")
        opt_mol = self.set_filter(opt_mol, n_verifies)
        for verifier in self.verifiers:
            verifier(opt_mol, multiplicity=self.multiplicity, save_dir=self.save_dir)

        # run torsional sampling
        if self.sampler:
            self.logger.info("Running torsional sampling...")
            energy_dict = opt_mol.energy
            KeepIDs = opt_mol.KeepIDs
            sorted_index = [k for k, v in sorted(energy_dict.items(), key = lambda item: item[1])] # Order by energy
            filter_index = [k for k in sorted_index if KeepIDs[k]][:n_sampling]
            found_lower_energy_index = {i: False for i in range(opt_mol.GetNumConformers())}
            for id in filter_index:
                original_energy = opt_mol.energy[id]
                new_mol = self.sampler(opt_mol, id, save_dir=self.save_dir)
                new_energy = new_mol.energy[id]
                if new_energy < original_energy:
                    found_lower_energy_index[id] = True
            # save which stable species found conformer with lower energy via torsional sampling
            with open(os.path.join(self.save_dir, "sampler_check_ids.pkl"), "wb") as f:
                pickle.dump(found_lower_energy_index, f)

            # Doesn't find any lower energy conformer! Using original result...
            if not all(value is False for value in found_lower_energy_index.values()):
                opt_mol = new_mol

        self.logger.info("Running final modules...")
        if self.final_modules:
            opt_mol = self.set_filter(opt_mol, n_refines)
            save_dir = os.path.join(self.save_dir, "final_modules")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for module in self.final_modules:
                opt_mol = module(opt_mol, multiplicity=self.multiplicity, save_dir=save_dir, smiles=self.smiles)

        # save which stable species passed full workflow
        with open(os.path.join(self.save_dir, "workflow_check_ids.pkl"), "wb") as f:
            pickle.dump(opt_mol.KeepIDs, f)

        return opt_mol
