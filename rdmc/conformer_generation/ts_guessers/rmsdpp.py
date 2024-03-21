from typing import Optional

from rdmc.external.xtb_tools.opt import run_xtb_calc
from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser


class RMSDPPGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the RMSD-PP method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    _avail = True

    def __init__(self, track_stats: Optional[bool] = False):
        """
        Initialize the RMSD-PP initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(RMSDPPGuesser, self).__init__(track_stats)

    def generate_ts_guesses(
        self, mols, multiplicity: Optional[int] = None, save_dir: Optional[str] = None
    ):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to None.
            save_dir (Optional[str], optional): The path to save the results. Defaults to None.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        ts_guesses, used_rp_combos = [], []
        multiplicity = multiplicity or 1
        for r_mol, p_mol in mols:
            _, ts_guess = run_xtb_calc(
                (r_mol, p_mol), return_optmol=True, job="--path", uhf=multiplicity - 1
            )
            if ts_guess:
                ts_guesses.append(ts_guess)
                used_rp_combos.append((r_mol, p_mol))

        if len(ts_guesses) == 0:
            # TODO: Need to think about catching this in the upper level
            return None

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        [ts_mol.AddConformer(t.GetConformer(), assignId=True) for t in ts_guesses]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol)

        return ts_mol
