import os
import pickle
from typing import Optional
from time import time


from rdkit import Chem


class TSOptimizer:
    """
    The abstract class for TS optimizer.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, track_stats: Optional[bool] = False):
        """
        Initialize the TS optimizer.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_ts_guesses(self, mol: "RDKitMol", save_dir: str, **kwargs):
        """
        The abstract method for optimizing TS guesses. It will be implemented in actual classes.
        The method needs to take ``mol`` in ``RDKitMol`` and ``save_dir`` as ``str`` as input arguments, and
        return the optimized molecule as ``RDKitMol``.

        Args:
            mol (RDKitMol): The TS in RDKitMol object with geometries embedded as conformers.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        raise NotImplementedError

    def save_opt_mols(
        self,
        save_dir: str,
        opt_mol: "RDKitMol",
        keep_ids: dict,
        energies: dict,
    ):
        """
        Save the information of the optimized TS geometries into the directory.

        Args:
            save_dir (str): The path to the directory to save the results.
            opt_mol (RDKitMol): The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
            keep_ids (dict): Dictionary of which opts succeeded and which failed.
            energies (dict): Dictionary of energies for each conformer.
        """
        # Save optimized ts mols
        ts_path = os.path.join(save_dir, "ts_optimized_confs.sdf")
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(ts_path)
            for i in range(opt_mol.GetNumConformers()):
                opt_mol.SetProp("Energy", str(energies[i]))
                ts_writer.write(opt_mol, confId=i)
        except Exception:
            raise
        finally:
            ts_writer.close()

        # save ids
        with open(os.path.join(save_dir, "opt_check_ids.pkl"), "wb") as f:
            pickle.dump(keep_ids, f)

    def __call__(self, mol: "RDKitMol", save_dir: Optional[str] = None, **kwargs):
        """
        Run the workflow to generate optimize TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            save_dir (str, optional): The path to save results. Defaults to ``None``.

        Returns:
            'RDKitMol': The optimized TS molecule as ``RDKitMol`` with 3D geometries embedded.
        """
        time_start = time()

        if not save_dir:
            save_dir = os.getcwd()

        opt_mol = self.optimize_ts_guesses(mol=mol, save_dir=save_dir, **kwargs)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return opt_mol
