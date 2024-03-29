from typing import Optional

from rdmc.conformer_generation.verifiers.base import FreqVerifier
from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.task.xtb import XTBTask


class XTBFrequencyVerifier(
    XTBTask,
    FreqVerifier,
):
    """
    The class for verifying the stable species by calculating and checking its frequencies using XTB.

    Args:
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations.
                                            Defaults to ``0.0`` cm-1.
        track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
    """

    def __init__(
        self,
        method: str = "GFN2-xTB",
        cutoff_frequency: Optional[float] = None,
        track_stats: bool = False,
    ):
        super(XTBFrequencyVerifier, self).__init__(method=method)
        super(XTBTask, self).__init__(cutoff_frequency, track_stats)

    def calc_freq(
        self,
        mol: "RDKitMol",
        conf_id: int,
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
        props = run_xtb_calc(
            mol, confId=conf_id, job="--hess", method=self.method, uhf=multiplicity - 1
        )
        return props["frequencies"]


XTBFreqVerifier = XTBFrequencyVerifier
