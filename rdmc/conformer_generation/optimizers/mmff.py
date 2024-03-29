import numpy as np

from rdmc.forcefield import RDKitFF
from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer


class MMFFOptimizer(ConfGenOptimizer):
    """
    Optimizer using the MMFF force field.

    Args:
        method (str, optional): The method to be used for stable species optimization. Defaults to ``"rdkit"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, method: str = "rdkit", track_stats: bool = False):
        super().__init__(track_stats)
        if method == "rdkit":
            self.ff = RDKitFF()
        elif method == "openbabel":
            raise NotImplementedError

    def is_available(self):
        return True

    def run(self, mol, **kwargs):
        """
        Optimize the conformers.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.

        Returns:
            RDKitMol: The optimized molecule as RDKitMol with 3D geometries embedded.
        """

        opt_mol = mol.Copy(copy_attrs=["KeepIDs"])

        self.ff.setup(opt_mol)
        results = self.ff.optimize_confs()
        return_codes, energies = zip(*results)  # kcal/mol
        opt_mol = self.ff.get_optimized_mol()

        for i, return_code in enumerate(return_codes):
            opt_mol.KeepIDs[i] = opt_mol.KeepIDs[i] and (return_code == 0)
        opt_mol.energy = {i: energy for i, energy in enumerate(energies)}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}

        if self.track_stats:
            self.n_failures = np.sum([r[0] == 1 for r in results])
            self.percent_failures = self.n_failures / len(results) * 100

        return opt_mol
