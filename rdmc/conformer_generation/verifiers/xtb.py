import os
import pickle

from rdmc.conformer_generation.verifiers.base import Verifier
from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.task.xtb import XTBTask


class XTBFrequencyVerifier(XTBTask, Verifier):
    """
    The class for verifying the stable species by calculating and checking its frequencies using XTB.

    Args:
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                            Defaults to ``-100.`` cm-1.
        track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
    """

    allowed_num_neg_freqs = 0

    def __init__(self, cutoff_frequency: float = 0.0, track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency used to determine if the molecules has an imaginary frequency mode.
                Defaults to ``0.0`` cm-1.
            track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
        """
        super().__init__(track_stats)
        self.cutoff_frequency = cutoff_frequency

    def run(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
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

                    freq_check = (
                        sum(frequencies < self.cutoff_frequency)
                        == self.allowed_num_neg_freqs
                    )
                    mol.KeepIDs[i] = freq_check

        if self.save_dir:
            with open(os.path.join(self.save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(mol.KeepIDs, f)

        return mol
