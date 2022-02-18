#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for providing initial guess geometries
"""

from rdmc import RDKitMol
import os.path as osp
import yaml
import numpy as np
from time import time
import torch
from torch_geometric.data import Batch
from .utils import *

try:
    from rdmc.external.GeoMol.model.model import GeoMol
    from rdmc.external.GeoMol.model.featurization import featurize_mol_from_smiles
    from rdmc.external.GeoMol.model.inference import construct_conformers
except ImportError:
    print("No GeoMol installation detected. Skipping import...")


class ConfGenEmbedder:
    def __init__(self, track_stats=False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []

    def embed_conformers(self, smiles, n_conformers):
        raise NotImplementedError

    def __call__(self, smiles, n_conformers):

        self.iter += 1
        time_start = time()
        mol_data = self.embed_conformers(smiles, n_conformers)

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {"iter": self.iter,
                 "time": time_end - time_start,
                 "n_success": self.n_success,
                 "percent_success": self.percent_success}
        self.stats.append(stats)
        return mol_data


class GeoMolEmbedder(ConfGenEmbedder):
    def __init__(self, trained_model_dir, track_stats=False):
        super(GeoMolEmbedder, self).__init__(track_stats)

        # TODO: add option of pre-pruning geometries using alpha values
        # TODO: inverstigate option of changing "temperature" each iteration to sample diverse geometries

        with open(osp.join(trained_model_dir, "model_parameters.yml")) as f:
            model_parameters = yaml.full_load(f)
        model = GeoMol(**model_parameters)

        state_dict = torch.load(osp.join(trained_model_dir, "best_model.pt"), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model
        self.tg_data = None
        self.std = model_parameters["hyperparams"]["random_vec_std"]

    def embed_conformers(self, smiles, n_conformers):

        # set "temperature"
        self.model.random_vec_std = self.std * (1 + self.iter / 10)

        # featurize data and run GeoMol
        if self.tg_data is None:
            self.tg_data = featurize_mol_from_smiles(smiles, dataset="drugs")
        data = Batch.from_data_list([self.tg_data])  # need to run this bc of dumb internal GeoMol processing
        self.model(data, inference=True, n_model_confs=n_conformers)

        # process predictions
        n_atoms = self.tg_data.x.size(0)
        model_coords = construct_conformers(data, self.model).double().cpu().detach().numpy()
        split_model_coords = np.split(model_coords, n_conformers, axis=1)

        # package in mol and return
        mol = RDKitMol.FromSmiles(smiles)
        mol.EmbedMultipleNullConfs(n=n_conformers)
        mol_data = []
        for i, x in enumerate(split_model_coords):
            conf = mol.Copy().GetConformer(i)
            positions = x.squeeze(axis=1)
            conf.SetPositions(positions)
            mol_data.append({"positions": positions,
                             "conf": conf,
                             "iter": self.iter})

        n_success = len(mol_data)
        self.n_success = n_success
        self.percent_success = n_success / n_conformers * 100

        return mol_data


class ETKDGEmbedder(ConfGenEmbedder):
    def __init__(self, track_stats=False):
        super(ETKDGEmbedder, self).__init__(track_stats)

        self.mol = None

    def embed_conformers(self, smiles, n_conformers):
        if self.mol is None:
            self.mol = RDKitMol.FromSmiles(smiles)

        mol = self.mol.Copy()
        mol.EmbedMultipleConfs(n_conformers)
        mol_data = mol_to_dict(mol, iter=self.iter)

        n_success = mol.GetNumConformers()
        self.n_success = n_success
        self.percent_success = n_success / n_conformers * 100

        return mol_data


class RandomEmbedder(ConfGenEmbedder):
    def __init__(self, track_stats=False):
        super(RandomEmbedder, self).__init__(track_stats)

        self.mol = None

    def embed_conformers(self, smiles, n_conformers):
        if self.mol is None:
            self.mol = RDKitMol.FromSmiles(smiles)

        mol = self.mol.Copy()
        mol.EmbedMultipleNullConfs(n_conformers)

        n_success = mol.GetNumConformers()
        self.n_success = n_success
        self.percent_success = n_success / n_conformers * 100

        return mol_to_dict(mol, iter=self.iter)
