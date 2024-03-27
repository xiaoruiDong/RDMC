from rdmc.conformer_generation.verifiers.xtb import (
    XTBFrequencyVerifier as XTBFreqVerifier,
)


class XTBFrequencyVerifier(XTBFreqVerifier):
    """
    The class for verifying the TS by calculating and checking its frequencies using XTB.

    Args:
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                            Defaults to ``-100.`` cm-1
        track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
    """

    allowed_num_neg_freqs = 1

    def __init__(self, cutoff_frequency: float = -10.0, track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                                                Defaults to ``-10.`` cm-1
            track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
        """
        super().__init__(cutoff_frequency, track_stats)
