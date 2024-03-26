import os
import pickle
from typing import Optional

from rdmc.conformer_generation.verifiers.base import Verifier
from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.comp_env import xtb_available


class XTBFrequencyVerifier(Verifier):
    """
    The class for verifying the stable species by calculating and checking its frequencies using XTB.

    Args:
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                            Defaults to ``-100.`` cm-1.
        track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
    """

    _avail = xtb_available

    def __init__(self, cutoff_frequency: float = -100.0, track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                                Defaults to ``-100.`` cm-1.
            track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
        """
        super(XTBFrequencyVerifier, self).__init__(track_stats)

        self.cutoff_frequency = cutoff_frequency

    def verify_guesses(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Verifying stable species guesses (or optimized stable species geometries).

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to ``1``.
            save_dir (_type_, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        if mol.GetNumAtoms() != 1:
            for i in range(mol.GetNumConformers()):
                if mol.KeepIDs[i]:
                    if mol.frequency[i] is None:
                        props = run_xtb_calc(
                            mol, confId=i, job="--hess", uhf=multiplicity - 1
                        )
                        frequencies = props["frequencies"]
                    else:
                        frequencies = mol.frequency[i]

                    freq_check = sum(frequencies < self.cutoff_frequency) == 0
                    mol.KeepIDs[i] = freq_check

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(mol.KeepIDs, f)

        return mol
