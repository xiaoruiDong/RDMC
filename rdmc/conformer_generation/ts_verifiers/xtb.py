import os
import pickle

from rdmc.conformer_generation.ts_verifiers.base import TSVerifier

from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc


class XTBFrequencyVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its frequencies using XTB.

    Args:
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                            Defaults to ``-100.`` cm-1
        track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
    """

    def __init__(self, cutoff_frequency: float = -100.0, track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                                Defaults to ``-100.`` cm-1
            track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
        """
        super(XTBFrequencyVerifier, self).__init__(track_stats)

        self.cutoff_frequency = cutoff_frequency

    def verify_ts_guesses(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> "RDKitMol":
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (str, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:
                if ts_mol.frequency[i] is None:
                    props = run_xtb_calc(
                        ts_mol, confId=i, job="--hess", uhf=multiplicity - 1
                    )
                    frequencies = props["frequencies"]
                else:
                    frequencies = ts_mol.frequency[i]

                # Check if the number of large negative frequencies is equal to 1
                freq_check = sum(frequencies < self.cutoff_frequency) == 1
                ts_mol.KeepIDs[i] = freq_check

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
