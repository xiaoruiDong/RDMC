from typing import Optional

from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.comp_env import xtb_available
from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser


class RMSDPPGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the RMSD-PP method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def is_available(self):
        """
        Check if the RMSD-PP method is available.

        Returns:
            bool: ``True`` if the RMSD-PP method is available, ``False`` otherwise.
        """
        return xtb_available

    def generate_ts_guess(
        self, rmol, pmol, multiplicity: Optional[int] = None, **kwargs
    ):
        """
        Generate a single TS guess.

        Args:
            rmol (RDKitMol): The reactant molecule in RDKitMol with 3D conformer saved with the molecule.
            pmol (RDKitMol): The product molecule in RDKitMol with 3D conformer saved with the molecule.
            multiplicity(int, optional): The multiplicity of the molecule. Defaults to 1.

        Returns:
            Tuple[np.ndarray, bool]: The generated guess positions and the success status.
        """
        multiplicity = multiplicity if multiplicity is not None else 1

        _, ts_guess = run_xtb_calc(
            (rmol, pmol),
            return_optmol=True,
            job="--path",
            uhf=multiplicity,
        )

        if ts_guess:
            return ts_guess.GetPositions(), True
        return None, False
