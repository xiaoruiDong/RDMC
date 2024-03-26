from typing import Optional
from time import time


class Verifier:
    """
    The abstract class for verifiers.

    Args:
        track_stats (bool, optional): Whether to track status. Defaults to ``False``.
    """

    def __init__(self, track_stats: bool = False):
        """
        Initialize the verifier.

        Args:
            track_stats (bool, optional): Whether to track status. Defaults to ``False``.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_guesses(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        The abstract method for verifying guesses (or optimized stable species geometries). The method need to take
        ``mol`` in RDKitMol, ``keep_ids`` in list, ``multiplicity`` in int, and ``save_dir`` in str, and returns
        a ``list`` indicating the ones passing the check.

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to ``1``.
            save_dir (_type_, optional): The directory path to save the results. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def __call__(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Run the workflow for verifying the stable species guessers (or optimized stable species conformers).

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the stable species. Defaults to ``1``.
            save_dir (_type_, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            list: a list of ``True`` and ``False`` indicating whether a conformer passes the check.
        """
        time_start = time()
        mol = self.verify_guesses(
            mol=mol, multiplicity=multiplicity, save_dir=save_dir, **kwargs
        )

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return mol
