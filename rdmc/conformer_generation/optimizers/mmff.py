from typing import List

import numpy as np

from rdmc.forcefield import RDKitFF
from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.utils import dict_to_mol


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

    def optimize_conformers(
        self,
        mol_data: List[dict],
    ) -> List[dict]:
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers sorted by energy.
        """
        if len(mol_data) == 0:
            return mol_data

        # Everytime calling dict_to_mol create a new molecule object
        # No need to Copy the molecule object in this function
        mol = dict_to_mol(mol_data)
        self.ff.setup(mol)
        results = self.ff.optimize_confs()
        _, energies = zip(*results)  # kcal/mol
        opt_mol = self.ff.get_optimized_mol()

        for c_id, energy in zip(range(len(mol_data)), energies):
            conf = opt_mol.GetEditableConformer(c_id)
            positions = conf.GetPositions()
            mol_data[c_id].update(
                {
                    "positions": positions,  # issues if not all opts succeeded?
                    "conf": conf,  # all confs share the same owning molecule `opt_mol`
                    "energy": energy,
                }
            )

        if self.track_stats:
            self.n_failures = np.sum([r[0] == 1 for r in results])
            self.percent_failures = self.n_failures / len(mol_data) * 100

        return sorted(mol_data, key=lambda x: x["energy"])
