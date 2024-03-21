from time import time
from typing import Optional


class TSVerifier:
    """
    The abstract class for TS verifiers.

    Args:
        track_stats (bool, optional): Whether to track status. Defaults to ``False``.
    """

    def __init__(self, track_stats: bool = False):
        """
        Initialize the TS verifier.

        Args:
            track_stats (bool, optional): Whether to track status. Defaults to ``False``.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_ts_guesses(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        The abstract method for verifying TS guesses (or optimized TS geometries). The method need to take
        ``ts_mol`` in ``RDKitMol``, ``keep_ids`` in ``list``, ``multiplicity`` in ``int``, and ``save_dir`` in ``str``.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (str, optional): The directory path to save the results. Defaults to ``None``.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def __call__(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> "RDKitMol":
        """
        Run the workflow for verifying the TS guessers (or optimized TS conformers).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (str, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        time_start = time()
        ts_mol = self.verify_ts_guesses(
            ts_mol=ts_mol, multiplicity=multiplicity, save_dir=save_dir, **kwargs
        )

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return ts_mol
