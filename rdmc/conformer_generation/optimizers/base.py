from time import time
from typing import List

from rdmc.conformer_generation.task.basetask import BaseTask


class ConfGenOptimizer(BaseTask):
    """
    Base class for the geometry optimizers used in conformer generation.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(self, track_stats: bool = False):

        super().__init__(track_stats)
        self.iter = 0
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None

    def optimize_conformers(self, mol_data: List[dict]):
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Raises:
            NotImplementedError: This function should be implemented in the child class.
        """
        raise NotImplementedError

    def __call__(
        self,
        mol_data: List[dict],
    ) -> List[dict]:
        """
        Run the workflow to optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers.
        """

        self.iter += 1
        time_start = time()
        mol_data = self.optimize_conformers(mol_data)

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {
            "iter": self.iter,
            "time": time_end - time_start,
            "n_failures": self.n_failures,
            "percent_failures": self.percent_failures,
            "n_opt_cycles": self.n_opt_cycles,
        }
        self.stats.append(stats)
        return mol_data
