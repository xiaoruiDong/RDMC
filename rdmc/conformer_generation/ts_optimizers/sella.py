import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.comp_env.sella import run_sella_opt, sella_available


class SellaOptimizer(ConfGenOptimizer):
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

    path_prefix = "sella_opt"

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
        super().__init__(track_stats)

        self.method = method
        self.fmax = fmax
        self.steps = steps

    def is_available(self):
        return sella_available

    def run_opt(self, mol: "RDKitMol", conf_id: int, **kwargs):
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            conf_id (int): The index of the TS guess to optimize.

        Returns:
            tuple: pos, success, energy, freq
        """
        work_dir = self.work_dir / f"{self.path_prefix}{conf_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        return run_sella_opt(
            mol,
            conf_id=conf_id,
            method=self.method,
            fmax=self.fmax,
            steps=self.steps,
            save_dir=work_dir,
        )
