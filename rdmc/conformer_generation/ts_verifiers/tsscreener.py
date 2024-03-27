import os
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
        rxn_smiles: str,
        **kwargs,
    ) -> "RDKitMol":
        """
        Screen poor TS guesses by using reacting mode from frequency calculation.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            rxn_smiles: The reaction SMILES.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        mol_data, ids = [], []

        # parse all optimization folders (which hold the frequency jobs)
        # Currently harded coded to get freq from opt and asssume opt are in save_dir
        log_dirs = sorted(
            [d for d in self.save_dir.glob("*opt*") if d.is_dir()],
            key=lambda d: int(d.name.split("opt")[-1]),
        )
        for log_dir in log_dirs:

            idx = int(log_dir.name.split("opt")[-1])
            if ts_mol.KeepIDs[idx]:

                log_path = next(log_dir.glob("*opt.log"))
                ts_freq_mol = convert_log_to_mol(log_path)

                if ts_freq_mol is None:
                    ts_mol.KeepIDs[idx] = False
                    continue

                # TSScreener needs rxn_smiles to prepare CGR
                ts_freq_mol.SetProp("Name", rxn_smiles)
                # Convert RDKitMol to TS-Screener input
                data = mol2data(ts_freq_mol, self.module.config, eval_mode=True)

                mol_data.append(data)
                ids.append(idx)

        if len(mol_data) == 0:
            return ts_mol

        # create data batch and run screener model
        batch_data = Batch.from_data_list(mol_data)
        preds = self.module.model(batch_data) > self.threshold

        # update which TSs to keep
        for idx, pred in zip(ids, preds):
            ts_mol.KeepIDs[ids] = bool(pred.item())

        # write ids to file
        with open(self.save_dir / "screener_check_ids.pkl", "wb") as f:
            pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
