import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser


_ts_egnn_avail = True


try:
    import torch
    from torch_geometric.data import Batch
    from ts_ml.trainers.ts_egnn_trainer import LitTSModule
    from ts_ml.dataloaders.ts_egnn_loader import TSDataset

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

except ImportError:
    _ts_egnn_avail = False
    print("No TS-EGNN installation detected. Skipping import...")


class TSEGNNGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the TS-EGNN model.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    _avail = _ts_egnn_avail

    def __init__(self, trained_model_dir: str, track_stats: Optional[bool] = False):
        """
        Initialize the TS-EGNN guesser.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(TSEGNNGuesser, self).__init__(track_stats)

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
        self.test_dataset = EvalTSDataset(self.config)

    def generate_ts_guesses(
        self,
        mols: list,
        multiplicity: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in ``RDKitMol`` with 3D conformer saved with the molecule.
        """
        # Generate the input for the TS-EGNN model
        rp_inputs = [
            (x[0], None, x[1]) for x in mols
        ]  # reactant, None (for TS), product
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # Use TS-EGNN to make initial guesses
        predicted_ts_coords = (
            self.module.model(batch_data)[:, :3].cpu().detach().numpy()
        )
        predicted_ts_coords = np.array_split(predicted_ts_coords, len(rp_inputs))

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(rp_inputs))
        [
            ts_mol.GetEditableConformer(i).SetPositions(
                np.array(predicted_ts_coords[i], dtype=float)
            )
            for i in range(len(rp_inputs))
        ]

        if save_dir:
            self.save_guesses(save_dir, mols, ts_mol)

        return ts_mol