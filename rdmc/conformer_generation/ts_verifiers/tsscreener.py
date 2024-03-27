import os
from glob import glob
from typing import Optional
import pickle

from rdmc import RDKitMol

from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.comp_env.pyg import Batch
from rdmc.conformer_generation.comp_env.ts_ml import LitScreenerModule, mol2data
from rdmc.conformer_generation.comp_env.software import package_available
from rdmc.conformer_generation.utils import convert_log_to_mol


class TSScreener(BaseTask):
    """
    The class for screening TS guesses using graph neural networks.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-Screener model.
        threshold (float): Threshold prediction at which we classify a failure/success. Defaults to ``0.95``.
        track_stats (bool, optional): Whether to track timing stats. Defaults to ``False``.
    """

    def __init__(
        self,
        trained_model_dir: str,
        threshold: float = 0.95,
        track_stats: bool = False,
    ):
        """
        Initialize the TS-Screener model.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-Screener model.
            threshold (float): Threshold prediction at which we classify a failure/success. Defaults to ``0.95``.
            track_stats (bool, optional): Whether to track timing stats. Defaults to ``False``.
        """
        super().__init__(track_stats)

        # Load the TS-Screener model
        self.module = LitScreenerModule.load_from_checkpoint(
            checkpoint_path=os.path.join(trained_model_dir, "best_model.ckpt")
        )

        # Setup configuration
        self.config = self.module.config
        self.module.model.eval()
        self.threshold = threshold

    def is_available(self) -> bool:
        """
        Check if the TS-Screener model is available.

        Returns:
            bool: Whether the TS-Screener model is available.
        """
        return package_available["TS-ML"]

    def run(
        self,
        ts_mol: "RDKitMol",
        **kwargs,
    ) -> "RDKitMol":
        """
        Screen poor TS guesses by using reacting mode from frequency calculation.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (str, optional): The directory path to save the results. Defaults to None.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        rxn_smiles = kwargs["rxn_smiles"]
        mol_data, ids = [], []

        # parse all optimization folders (which hold the frequency jobs)
        for log_dir in sorted(
            [d for d in glob(os.path.join(self.save_dir, "*opt*")) if os.path.isdir(d)],
            key=lambda x: int(x.split("opt")[-1]),
        ):

            idx = int(log_dir.split("opt")[-1])
            if ts_mol.KeepIDs[idx]:
                freq_log_path = glob(os.path.join(log_dir, "*opt.log"))[0]
                ts_freq_mol = convert_log_to_mol(freq_log_path)

                if ts_freq_mol is None:
                    ts_mol.KeepIDs.update({idx: False})
                    continue

                ts_freq_mol.SetProp("Name", rxn_smiles)
                data = mol2data(ts_freq_mol, self.module.config, eval_mode=True)

                mol_data.append(data)
                ids.append(idx)

        # return if nothing to screen
        if len(mol_data) == 0:
            return

        # create data batch and run screener model
        batch_data = Batch.from_data_list(mol_data)
        preds = self.module.model(batch_data) > self.threshold

        # update which TSs to keep
        updated_keep_ids = {idx: pred.item() for idx, pred in zip(ids, preds)}
        ts_mol.KeepIDs.update(updated_keep_ids)

        # write ids to file
        with open(os.path.join(self.save_dir, "screener_check_ids.pkl"), "wb") as f:
            pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
