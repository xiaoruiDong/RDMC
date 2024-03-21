from pathlib import Path

import numpy as np
from rdmc.conformer_generation.embedders.base import ConfGenEmbedder

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
        device: str = "cpu",
    ):
        if GeoMol is None:
            raise ImportError(
                "No GeoMol installation detected. Please install the GeoMol fork at https://github.com/xiaoruiDong/GeoMol."
            )
        super(GeoMolEmbedder, self).__init__(track_stats)

        # TODO: add option of pre-pruning geometries using alpha values
        # TODO: investigate option of changing "temperature" each iteration to sample diverse geometries
        self.device = device

        trained_model_dir = (
            geomol_model_path / dataset
            if trained_model_dir is None
            else Path(trained_model_dir)
        )
        with open(trained_model_dir / "model_parameters.yml") as f:
            model_parameters = yaml.full_load(f)
        model = GeoMol(**model_parameters)

        state_dict = torch.load(
            trained_model_dir / "best_model.pt", map_location=torch.device(device)
        )
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

    def embed_conformers(self, n_conformers: int):
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
        data = from_data_list([self.tg_data]).to(
            self.device
        )  # need to run this bc of dumb internal GeoMol processing
        self.model(data, inference=True, n_model_confs=n_conformers)

        # process predictions
        model_coords = (
            construct_conformers(data, self.model, self.device)
            .double()
            .cpu()
            .detach()
            .numpy()
        )
        split_model_coords = np.split(model_coords, n_conformers, axis=1)

        # package in mol and return
        self.mol.EmbedMultipleNullConfs(n=n_conformers, random=False)
        for i, x in enumerate(split_model_coords):
            conf = self.mol.GetEditableConformer(i)
            conf.SetPositions(x.squeeze(axis=1))
        return self.mol