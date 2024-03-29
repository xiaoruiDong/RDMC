from abc import abstractmethod
from time import time
from typing import List, Optional

from rdmc.conformer_generation.task.basetask import BaseTask


class ConfGenPruner(BaseTask):
    """
    Base class for conformer pruning.

    Args:
        track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
    """

    def __init__(self, track_stats: bool = False):

        super().__init__(track_stats=track_stats)
        self.iter = 0
        self.n_input_confs = None
        self.n_pruned_confs = None
        self.n_output_confs = None

    @abstractmethod
    def run(
        self,
        current_mol_data: List[dict],
        unique_mol_data: Optional[List[dict]] = None,
        sort_by_energy: bool = True,
        return_ids: bool = False,
    ):
        """
        Prune conformers.

        Args:
            current_mol_data (List[dict]): conformer data of the current iteration.
            unique_mol_data (List[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """

    def __call__(
        self,
        current_mol_data: List[dict],
        unique_mol_data: Optional[List[dict]] = None,
        sort_by_energy: bool = True,
        return_ids: bool = False,
    ):
        """
        Execute the task of pruning conformers.

        Args:
            current_mol_data (List[dict]): conformer data of the current iteration.
            unique_mol_data (List[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Returns:
            List[dict]: Updated conformer data.
        """
        self.iter += 1
        time_start = time()
        mol_data = self.run(
            current_mol_data, unique_mol_data, sort_by_energy, return_ids
        )

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {
            "iter": self.iter,
            "time": time_end - time_start,
            "n_input_confs": self.n_input_confs,
            "n_pruned_confs": self.n_pruned_confs,
            "n_output_confs": self.n_output_confs,
        }
        self.stats.append(stats)
        return mol_data
