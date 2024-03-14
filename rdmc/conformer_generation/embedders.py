#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing initial guess geometries
"""

from pathlib import Path

from rdmc import RDKitMol
import os.path as osp

import numpy as np
from time import time

try:
    import torch
    from torch_geometric.data import Batch
except ImportError:
    pass

from .utils import *

try:
    import torch
    from geomol.model import GeoMol
    from geomol.featurization import featurize_mol_from_smiles, from_data_list
    from geomol.inference import construct_conformers
    from geomol.utils import model_path as geomol_model_path
    import yaml  # only used to load GeoMol parameters
except ImportError as e:
    GeoMol = None
    print(e)
    print("No GeoMol installation detected. Skipping import...")
    print("Please install the GeoMol fork at https://github.com/xiaoruiDong/GeoMol")


class ConfGenEmbedder:
    """
    Base class for conformer generation embedders.
    """

    def __init__(self, track_stats=False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []
        self.smiles = None

    def update_mol(self, smiles: str):
        """
        Update the molecule graph based on the SMILES string.

        Args:
            smiles (str): SMILES string of the molecule
        """
        # Only update the molecule if smiles is changed
        # Only copy the molecule graph from the previous run rather than conformers
        if smiles != self.smiles:
            self.smiles = smiles
            self.mol = RDKitMol.FromSmiles(smiles)
        else:
            # Copy the graph but remove conformers
            self.mol = self.mol.Copy(quickCopy=True)

    def embed_conformers(
        self,
        n_conformers: int
    ):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def update_stats(
        self,
        n_trials: int,
        time: float = 0.
    ) -> dict:
        """
        Update the statistics of the conformer generation.

        Args:
            n_trials (int): Number of trials
            time (float, optional): Time spent on conformer generation. Defaults to ``0.``.

        Returns:
            dict: Statistics of the conformer generation
        """
        n_success = self.mol.GetNumConformers()
        self.n_success = n_success
        self.percent_success = n_success / n_trials * 100
        stats = {
            "iter": self.iter,
            "time": time,
            "n_success": self.n_success,
            "percent_success": self.percent_success
        }
        self.stats.append(stats)
        return stats

    def write_mol_data(self):
        """
        Write the molecule data.

        Returns:
            dict: Molecule data.
        """
        return mol_to_dict(self.mol, copy=False, iter=self.iter)

    def __call__(
        self,
        smiles: str,
        n_conformers: int
    ):
        """
        Embed conformers according to the molecule graph.

        Args:
            smiles (str): SMILES string of the molecule.
            n_conformers (int): Number of conformers to generate.

        Returns:
            dict: Molecule data.
        """
        self.iter += 1
        time_start = time()
        self.update_mol(smiles)
        self.embed_conformers(n_conformers)
        mol_data = self.write_mol_data()

        if not self.track_stats:
            return mol_data

        time_end = time()
        self.update_stats(n_trials=n_conformers,
                          time=time_end - time_start)
        return mol_data


class GeoMolEmbedder(ConfGenEmbedder):
    """
    Embed conformers using GeoMol.

    Args:
            trained_model_dir (str, optional): Directory of the trained model. If not provided, the models distributed with the package will be used.
            dataset (str, optional): Dataset used for training. Defaults to ``"drugs"``.
            temp_schedule (str, optional): Temperature schedule. Defaults to ``"linear"``.
            track_stats (bool, optional): Whether to track the statistics of the conformer generation. Defaults to ``False``.
    """

    def __init__(
        self,
        trained_model_dir: str = None,
        dataset: str = "drugs",
        temp_schedule: str = "linear",
        track_stats: bool = False,
        device: str = 'cpu',
    ):
        if GeoMol is None:
            raise ImportError("No GeoMol installation detected. Please install the GeoMol fork at https://github.com/xiaoruiDong/GeoMol.")
        super(GeoMolEmbedder, self).__init__(track_stats)

        # TODO: add option of pre-pruning geometries using alpha values
        # TODO: investigate option of changing "temperature" each iteration to sample diverse geometries
        self.device = device

        trained_model_dir = geomol_model_path / dataset if trained_model_dir is None else Path(trained_model_dir)
        with open(trained_model_dir / "model_parameters.yml") as f:
            model_parameters = yaml.full_load(f)
        model = GeoMol(**model_parameters)

        state_dict = torch.load(trained_model_dir / "best_model.pt", map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model
        self.tg_data = None
        self.std = model_parameters["hyperparams"]["random_vec_std"]
        self.temp_schedule = temp_schedule
        self.dataset = dataset

    def to(self, device: str):
        self.device = device
        self.model.to(device)

    def embed_conformers(
        self,
        n_conformers: int
    ):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        # set "temperature"
        if self.temp_schedule == "none":
            self.model.random_vec_std = self.std
        elif self.temp_schedule == "linear":
            self.model.random_vec_std = self.std * (1 + self.iter / 10)

        # featurize data and run GeoMol
        if self.tg_data is None:
            self.tg_data = featurize_mol_from_smiles(self.smiles, dataset=self.dataset)
        data = from_data_list([self.tg_data]).to(self.device)  # need to run this bc of dumb internal GeoMol processing
        self.model(data, inference=True, n_model_confs=n_conformers)

        # process predictions
        model_coords = construct_conformers(data, self.model, self.device).double().cpu().detach().numpy()
        split_model_coords = np.split(model_coords, n_conformers, axis=1)

        # package in mol and return
        self.mol.EmbedMultipleNullConfs(n=n_conformers, random=False)
        for i, x in enumerate(split_model_coords):
            conf = self.mol.GetConformer(i)
            conf.SetPositions(x.squeeze(axis=1))
        return self.mol


class ETKDGEmbedder(ConfGenEmbedder):
    """
    Embed conformers using ETKDG.
    """

    def embed_conformers(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        self.mol.EmbedMultipleConfs(n_conformers)
        return self.mol


class RandomEmbedder(ConfGenEmbedder):
    """
    Embed conformers with coordinates of random numbers.
    """

    def embed_conformers(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        self.mol.EmbedMultipleNullConfs(n_conformers, random=True)
        return self.mol
