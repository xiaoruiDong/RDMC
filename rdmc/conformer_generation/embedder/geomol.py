#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing molecule guess geometries by GeoMol
"""

import os.path as osp
import yaml

import numpy as np

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import SOFTWARE_AVAIL, timer

try:
    import torch
except ImportError:
    SOFTWARE_AVAIL["pytorch"] = False
    print("No torch installation detected. Skipping import...")
else:
    SOFTWARE_AVAIL["pytorch"] = True

try:
    from torch_geometric.data import Batch
except:
    SOFTWARE_AVAIL["torch_geometric"] = False
    print("No torch_geometric installation detected. Skipping import...")
else:
    SOFTWARE_AVAIL["torch_geometric"] = True

try:
    from geomol.model import GeoMol
    from geomol.featurization import featurize_mol, from_data_list
    from geomol.inference import construct_conformers
except ImportError:
    print("No external GeoMol installation detected. Importing from RDMC.external.GeoMol...")
    try:
        from rdmc.external.GeoMol.geomol.model import GeoMol
        from rdmc.external.GeoMol.geomol.featurization import featurize_mol, from_data_list
        from rdmc.external.GeoMol.geomol.inference import construct_conformers
    except ImportError:
        SOFTWARE_AVAIL["geomol"] = False
        print("No GeoMol installation detected. Skipping import...")
    else:
        SOFTWARE_AVAIL["geomol"] = True
else:
    SOFTWARE_AVAIL["geomol"] = True


class GeoMolEmbedder(Task):

    request_external_software = ['pytorch', 'torch_geometric', 'geomol']

    def __init__(self,
                 model_dir: str,
                 dataset: str = "drugs",
                 temp_schedule: str = "linear",
                 **kwargs,
                 ):
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
        super().__init__(model_dir=model_dir,
                         dataset=dataset,
                         temp_schedule=temp_schedule,
                         **kwargs)

        # TODO: add option of pre-pruning geometries using alpha values
        # TODO: investigate option of changing "temperature" each iteration to sample diverse geometries

    def check_external_software(self):
        return all([SOFTWARE_AVAIL[software] for software in self.request_external_software])

    def task_prep(self,
                  model_dir,
                  dataset="drugs",
                  temp_schedule="linear",
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
        self.tg_data = None

    @timer
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
        # featurize data and run GeoMol
        if self.tg_data is None:
            self.tg_data = featurize_mol(
                                mol=mol._mol,
                                dataset=self.dataset)
        # need to run this bc of dumb internal GeoMol processing
        data = from_data_list([self.tg_data])
        self.model(data,
                   inference=True,
                   n_model_confs=n_conformers)

        # process predictions
        model_coords = construct_conformers(data, self.model).double().cpu().detach().numpy()
        split_model_coords = np.split(model_coords, n_conformers, axis=1)

        # package in mol and return
        mol.EmbedMultipleNullConfs(n=n_conformers, random=False)
        for i, x in enumerate(split_model_coords):
            mol.SetPositions(x.squeeze(axis=1), i)

        self.n_success = n_conformers

        return mol
