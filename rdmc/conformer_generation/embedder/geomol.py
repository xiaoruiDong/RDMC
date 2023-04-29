#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing molecule guess geometries by GeoMol
"""

from functools import lru_cache
import os.path as osp
import yaml

import numpy as np

from rdmc.conformer_generation.embedder.conformer import ConformerEmbedder
from rdmc.conformer_generation.utils import register_software

with register_software('pytorch'):
    import torch

with register_software('geomol'):
    try:
        from geomol.model import GeoMol
        from geomol.featurization import featurize_mol, from_data_list
        from geomol.inference import construct_conformers
    except ImportError:
        from rdmc.external.GeoMol.geomol.model import GeoMol
        from rdmc.external.GeoMol.geomol.featurization import featurize_mol, from_data_list
        from rdmc.external.GeoMol.geomol.inference import construct_conformers


class GeoMolEmbedder(ConformerEmbedder):
    """
    Generate conformers using GeoMol.

    Args:
        model_dir (str): Path to directory containing trained model. Within the path, there should be a
                        "best_model.pt" file and a "model_parameters.yml" file.
        dataset (str): Dataset used to train the model with two options: "drugs" or "qm9".
                        It influences how molecules are featurized. "qm9" only supports CHNOF molecules,
                        while "drugs" have more elements supported. Defaults to "drugs".
        temp_schedule (str): Temperature schedule for sampling conformers. Two options:
                            "linear" or "none".
    """

    request_external_software = ['pytorch', 'geomol']

    def task_prep(self,
                  model_dir: str,
                  dataset: str = "drugs",
                  temp_schedule: str = "linear",
                  **kwargs,):
        """
        Prepare the GeoMol task by loading models and setting parameters.
        """

        with open(osp.join(model_dir, "model_parameters.yml")) as f:
            self.model_parameters = yaml.full_load(f)
        self.model = GeoMol(**self.model_parameters)

        state_dict = torch.load(
                osp.join(model_dir, "best_model.pt"), map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=True)
        # set "temperature"
        self.model.random_vec_std = self.model_parameters["hyperparams"]["random_vec_std"]
        if temp_schedule == "linear":
            self.model.random_vec_std *= (1 + self.iter / 10)
        self.model.eval()
        self.dataset = dataset

    @lru_cache(maxsize=1)
    def get_tg_data(self,
                    mol: 'Chem.Mol',
                    dataset: str):
        """
        Get the molecule featurization in Torch Geometric Data.

        Args:
            mol (Mol): The molecule in RDKit Mol object.
            dataset (str): Dataset used to train the model with two options: "drugs" or "qm9".
        """
        return featurize_mol(mol=mol._mol,
                             dataset=self.dataset)

    @ConformerEmbedder.timer
    def run(self,
            mol: 'RDKitMol',
            n_conformers: int,
            **kwargs):
        """
        Generate conformers using GeoMol.

        Args:
            mol (RDKitMol): RDKit molecule object. Note, this function will overwrite the conformers of the molecule.
            n_conformers (int): Number of conformers to generate.
        """
        mol.EmbedMultipleNullConfs(n=n_conformers, random=False)
        mol.keep_ids = [False] * n_conformers

        # featurize data and run GeoMol
        tg_data = self.get_tg_data(mol=mol,
                                   dataset=self.dataset)
        # need to run this bc of dumb internal GeoMol processing
        try:
            data = from_data_list([tg_data])
        except TypeError:
            raise ValueError("GeoMol requires a molecule with at least one rotor.")
        self.model(data,
                   inference=True,
                   n_model_confs=n_conformers)

        # process predictions
        model_coords = construct_conformers(data, self.model).double().cpu().detach().numpy()
        split_model_coords = np.split(model_coords, n_conformers, axis=1)

        # package in mol and return
        for i, x in enumerate(split_model_coords):
            try:
                mol.SetPositions(x.squeeze(axis=1), i)
                mol.keep_ids[i] = True
            except Exception:
                pass

        return mol
