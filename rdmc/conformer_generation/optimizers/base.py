import pickle

import numpy as np
from rdkit import Chem

from rdmc.conformer_generation.task.basetask import BaseTask


class ConfGenOptimizer(BaseTask):
    """
    Base class for the geometry optimizers used in conformer generation.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, track_stats: bool = False):

        super().__init__(track_stats)
        self.iter = 0
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = []

    def run(self, mol: "RDKitMol", **kwargs):
        """
        Optimize the conformers.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.

        Returns:
            RDKitMol: The optimized molecule as RDKitMol with 3D geometries embedded.
        """

        opt_mol = mol.Copy(copy_attrs=["KeepIDs"])
        opt_mol.energy = {}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}

        n_opt_to_run = sum(opt_mol.KeepIDs.values())

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

        if self.track_stats:
            self.n_failures = n_opt_to_run - sum(opt_mol.KeepIDs.values())
            self.percent_failures = self.n_failures / n_opt_to_run * 100

        return opt_mol

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

    def update_stats(self, exe_time: float, mol, *args, **kwargs):
        """
        Update the statistics of the conformer generation.

        Args:
            exe_time (float): Execution time of the conformer generation
            n_conformers (int): Number of conformers planned to generate
        """
        stats = {
            "iter": self.iter,
            "time": exe_time,
            "n_failures": self.n_failures,
            "percent_failures": self.percent_failures,
            "n_opt_cycles": self.n_opt_cycles,
        }
        self.stats.append(stats)

    def __call__(self, *args, **kwargs):

        self.iter += 1
        return super().__call__(*args, **kwargs)

    def save_opt_mols(
        self,
        opt_mol: "RDKitMol",
    ):
        """
        Save the information of the optimized stable species into the directory.

        Args:
            opt_mol (RDKitMol): The optimized stable species in RDKitMol with 3D conformer saved with the molecule.
        """

        # Save optimized stable species mols
        path = self.save_dir / "optimized_confs.sdf"
        try:
            writer = Chem.rdmolfiles.SDWriter(str(path))
            for i in range(opt_mol.GetNumConformers()):
                opt_mol.SetDoubleProp("Energy", float(opt_mol.energy[i]))
                writer.write(opt_mol, confId=i)
        except Exception:
            raise
        finally:
            writer.close()

        # save ids
        with open(self.save_dir / "opt_check_ids.pkl", "wb") as f:
            pickle.dump(opt_mol.KeepIDs, f)
