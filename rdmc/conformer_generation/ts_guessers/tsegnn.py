import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser
from rdmc.conformer_generation.comp_env.pyg import Batch
from rdmc.conformer_generation.comp_env.torch import torch
from rdmc.conformer_generation.comp_env.ts_ml import LitTSModule, TSDataset
from rdmc.conformer_generation.comp_env.software import package_available


def get_test_dataset(config):

    class EvalTSDataset(TSDataset):
        def __init__(self, config):
            self.mols = []
            self.no_shuffle_mols = True  # randomize which is reactant/product
            self.no_mol_prep = False  # prep as if starting from SMILES
            self.set_similar_mols = (
                False  # use species (r/p) which is more similar to TS as starting mol
            )
            self.product_loss = False
            self.prod_feat = config[
                "prod_feat"
            ]  # whether product features include distance or adjacency

    return EvalTSDataset(config)


class TSEGNNGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the TS-EGNN model.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, trained_model_dir: str, track_stats: Optional[bool] = False):
        """
        Initialize the TS-EGNN guesser.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)

        # Load the TS-EGNN model
        checkpoint_path = os.path.join(trained_model_dir, "best_model.ckpt")
        model_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model_checkpoint["hyper_parameters"]["config"][
            "training"
        ] = False  # we're not training
        self.module = LitTSModule(model_checkpoint["hyper_parameters"]["config"])
        self.module.load_state_dict(state_dict=model_checkpoint["state_dict"])

        # Setup TS-EGNN configuration
        self.config = self.module.config
        self.module.model.eval()
        self.test_dataset = get_test_dataset(self.config)

    def is_available(self):
        """
        Check whether the TS-EGNN model is available.

        Returns:
            bool: Whether the TS-EGNN model is available.
        """
        return package_available["TS-ML"]

    def generate_ts_guesses(self, mols: list, **kwargs):
        """
        Generate TS guesses.

        Args:
            mols (list): A list of reactant and product pairs.

        Returns:
            Tuple[dict, dict]: The generated guesses positions and the success status. The keys are the conformer IDs. The values are the positions and the success status.
        """
        # Generate the input for the TS-EGNN model
        n_confs = len(mols)
        rp_inputs = [
            (x[0], None, x[1]) for x in mols
        ]  # reactant, None (for TS), product
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # Use TS-EGNN to make initial guesses
        predicted_ts_coords = (
            self.module.model(batch_data)[:, :3].cpu().detach().numpy()
        )
        predicted_ts_coords = np.array_split(predicted_ts_coords, n_confs)

        positions = {
            i: np.array(predicted_ts_coords[i], dtype=float) for i in range(n_confs)
        }
        successes = {i: True for i in range(n_confs)}
        return positions, successes
