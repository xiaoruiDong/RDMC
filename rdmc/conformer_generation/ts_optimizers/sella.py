import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
try:
    from rdmc.conformer_generation.comp_env.sella import run_sella_opt
except BaseException:
    print("No Sella installation deteced. Skipping import...")


class SellaOptimizer(TSOptimizer):
    """
    The class to optimize TS geometries using the Sella algorithm.
    It uses XTB as the backend calculator, ASE as the interface, and Sella module from the Sella repo.

    Args:
        method (str, optional): The method in XTB used to optimize the geometry. Options are
                                ``'GFN1-xTB'`` and ``'GFN2-xTB'``. Defaults to ``"GFN2-xTB"``.
        fmax (float, optional): The force threshold used in the optimization. Defaults to ``1e-3``.
        steps (int, optional): Max number of steps allowed in the optimization. Defaults to ``1000``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        method: str = "GFN2-xTB",
        fmax: float = 1e-3,
        steps: int = 1000,
        track_stats: bool = False,
    ):
        """
        Initiate the Sella optimizer.

        Args:
            method (str, optional): The method in XTB used to optimize the geometry. Options are 'GFN1-xTB' and 'GFN2-xTB'. Defaults to "GFN2-xTB".
            fmax (float, optional): The force threshold used in the optimization. Defaults to 1e-3.
            steps (int, optional): Max number of steps allowed in the optimization. Defaults to 1000.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(SellaOptimizer, self).__init__(track_stats)

        self.method = method
        self.fmax = fmax
        self.steps = steps

    def optimize_ts_guesses(
        self, mol: "RDKitMol", save_dir: Optional[str] = None, **kwargs
    ):
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            save_dir (str, optional): The path to save results. Defaults to ``None``.

        Returns:
            RDKitMol: The optimized TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        opt_mol = mol.Copy(copy_attrs=["KeepIDs"])
        opt_mol.energy = {}
        opt_mol.frequency = {i: None for i in range(mol.GetNumConformers())}
        for i in range(mol.GetNumConformers()):

            if not opt_mol.KeepIDs[i]:
                opt_mol.AddNullConformer(confId=i)
                opt_mol.energy.update({i: np.nan})
                continue

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"sella_opt{i}")
                os.makedirs(ts_conf_dir, exist_ok=True)

            opt_mol = run_sella_opt(
                opt_mol,
                method=self.method,
                confId=i,
                fmax=self.fmax,
                steps=self.steps,
                save_dir=ts_conf_dir,
                copy_attrs=["KeepIDs", "energy", "frequency"],
            )
        if save_dir:
            self.save_opt_mols(save_dir, opt_mol, opt_mol.KeepIDs, opt_mol.energy)

        return opt_mol
