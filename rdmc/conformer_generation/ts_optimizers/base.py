import pickle

from rdkit import Chem
import numpy as np

from rdmc.conformer_generation.task.basetask import BaseTask


class TSOptimizer(BaseTask):
    """
    The abstract class for TS optimizer.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def run(self, mol: "RDKitMol", **kwargs):
        """
        The abstract method for optimizing TS guesses. It will be implemented in actual classes.
        The method needs to take ``mol`` in ``RDKitMol`` and return the optimized molecule as
        ``RDKitMol``.

        Args:
            mol (RDKitMol): The TS in RDKitMol object with geometries embedded as conformers.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        opt_mol = mol.Copy(copy_attrs=["KeepIDs"])
        opt_mol.energy = {}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}

        for i in range(mol.GetNumConformers()):

            if not opt_mol.KeepIDs[i]:
                opt_mol.energy[i] = np.nan
                continue

            pos, success, energy, freq = self.run_opt(mol, conf_id=i, **kwargs)

            opt_mol.KeepIDs[i] = success
            opt_mol.energy[i] = energy
            opt_mol.frequency[i] = freq
            if pos is not None:
                opt_mol.SetPositions(pos, confId=i)

        if self.save_dir:
            self.save_opt_mols(opt_mol)

    def run_opt(self, mol: "RDKitMol", conf_id: int, **kwargs):
        """
        The abstract method for optimizing a single TS guess. It will be implemented in actual classes.
        The method needs to take ``mol`` in ``RDKitMol`` and ``conf_id`` as the index of the TS guess
        and return the optimized geometry as ``np.ndarray``.

        Args:
            mol (RDKitMol): The TS in RDKitMol object with geometries embedded as conformers.
            conf_id (int): The index of the TS guess.

        Returns:
            np.ndarray: The optimized geometry.
        """
        raise NotImplementedError()

    def save_opt_mols(self, opt_mol):
        """
        Save the information of the optimized TS geometries into the directory.

        Args:
            opt_mol (RDKitMol): The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        # Save optimized ts mols
        ts_path = self.save_dir / "ts_optimized_confs.sdf"
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(str(ts_path))
            for i in range(opt_mol.GetNumConformers()):
                opt_mol.SetDoubleProp("Energy", float(opt_mol.energy[i]))
                ts_writer.write(opt_mol, confId=i)
        except Exception:
            raise
        finally:
            ts_writer.close()

        # save ids
        with open(self.save_dir / "opt_check_ids.pkl", "wb") as f:
            pickle.dump(opt_mol.KeepIDs, f)
