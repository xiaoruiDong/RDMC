import numpy as np
from typing import Optional, Tuple

from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.task.xtb import XTBTask


class XTBOptimizer(ConfGenOptimizer, XTBTask):
    """
    Optimizer using the xTB.

    Args:
        method (str, optional): The method to be used for species optimization. Defaults to ``"gff"``.
        level (str, optional): The level of theory. Defaults to ``"normal"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self, method: str = "gff", level: str = "normal", track_stats: bool = False
    ):
        super().__init__(track_stats=track_stats)
        self.method = method
        self.level = level

    def run_opt(
        self,
        mol: "RDKitMol",
        conf_id: int,
        multiplicity: int = 1,
        **kwargs,
    ) -> Tuple[Optional["np.ndarray"], bool, float, Optional["np.ndarray"]]:
        """
        Optimize the conformer.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            conf_id (int): The conformer id.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.

        Returns:
            tuple: pos, success, energy, freq
        """
        pos, success, energy, freq = None, False, np.nan, None

        try:
            props, opt_mol = run_xtb_calc(
                mol,
                confId=conf_id,
                job="--opt",
                return_optmol=True,
                method=self.method,
                level=self.level,
                uhf=multiplicity - 1,
            )

        except ValueError as e:
            print(f"Run into error in conformer optimization.")
            print(e)
        else:
            pos = opt_mol.GetPositions()
            energy = props["total energy"]

        return pos, success, energy, freq
