import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser
from rdmc.conformer_generation.comp_env.torch import torch
from rdmc.conformer_generation.comp_env.pyg import Batch
from rdmc.conformer_generation.comp_env.ts_ml import LitTSGCNModule, TSGCNDataset
from rdmc.conformer_generation.comp_env.software import package_available


def get_test_dataset(config):
    """
    Get the test dataset for TS-GCN.

    Args:
        config (dict): The configuration of the TS-GCN model.

    Returns:
        torch.utils.data.Dataset: The test dataset for TS-GCN.
    """

    class EvalTSGCNDataset(TSGCNDataset):
        def __init__(self, config):
            self.no_shuffle_mols = True  # randomize which is reactant/product
            self.no_mol_prep = False  # prep as if starting from SMILES

    return EvalTSGCNDataset(config)


class TSGCNGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the TS-GCN model.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-GCN model.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, trained_model_dir: str, track_stats: bool = False):
        """
        Initialize the TS-EGNN guesser.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-GCN model.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)

        # Load the TS-GCN model
        self.module = LitTSGCNModule.load_from_checkpoint(
            checkpoint_path=os.path.join(trained_model_dir, "best_model.ckpt"),
            strict=False,  # TODO: make sure d_init can be properly loaded
        )

        # Set the configuration of TS-GCN
        self.config = self.module.config
        self.module.model.eval()
        self.config["shuffle_mols"] = False
        self.config["prep_mols"] = False  # ts_generator class takes care of prep
        self.test_dataset = get_test_dataset(self.config)

    def is_available(self):
        """
        Check if the TS-GCN model is available.

        Returns:
            bool: Whether the TS-GCN model is available.
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
        # Prepare the input for the TS-GCN model
        n_confs = len(mols)
        rp_inputs = [
            (x[0], None, x[1]) for x in mols
        ]  # reactant, None (for TS), product
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # Use TS-GCN to make initial guesses
        self.module.model(batch_data)
        predicted_ts_coords = torch.vstack(
            [
                c[: m[0].GetNumAtoms()]
                for c, m in zip(batch_data.coords, batch_data.mols)
            ]
        )
        predicted_ts_coords = np.array_split(
            predicted_ts_coords.cpu().detach().numpy(), n_confs
        )

        positions = {
            i: np.array(predicted_ts_coords[i], dtype=float) for i in range(n_confs)
        }
        successes = {i: True for i in range(n_confs)}
        return positions, successes
