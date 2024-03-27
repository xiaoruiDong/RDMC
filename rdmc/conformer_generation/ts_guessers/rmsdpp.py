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

    def run(
        self,
        mols,
        multiplicity: Optional[int] = None,
    ):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to None.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        ts_guesses, used_rp_combos = [], []
        multiplicity = multiplicity or 1
        for r_mol, p_mol in mols:
            used_rp_combos.append((r_mol, p_mol))

            _, ts_guess = run_xtb_calc(
                (r_mol, p_mol), return_optmol=True, job="--path", uhf=multiplicity - 1
            )

            if ts_guess:
                ts_guesses.append(ts_guess)
            else:
                ts_guesses.append(None)

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [
            ts_mol.GetEditableConformer(i).SetPositions(p)
            for i, p in enumerate(ts_guesses)
            if p is not None
        ]

        if self.save_dir:
            self.save_guesses(used_rp_combos, ts_mol)

        ts_mol.KeepIDs = {i: p is not None for i, p in enumerate(ts_guesses)}

        return ts_mol
