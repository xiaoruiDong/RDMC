from typing import List, Optional

from rdmc.conformer_generation.pruners.base import ConfGenPruner
from rdmc.conformer_generation.comp_env.xtb.crest import run_cre_check
from rdmc.conformer_generation.comp_env import crest_available


class CRESTPruner(ConfGenPruner):
    """
    Prune conformers using CREST.

    Args:
        ethr (float, optional): Energy threshold. Defaults to ``0.15``.
        rthr (float, optional): RMSD threshold. Defaults to ``0.125``.
        bthr (float, optional): Bond threshold. Defaults to ``0.01``.
        ewin (int, optional): Energy window. Defaults to ``10000``.
        track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
    """

    def __init__(
        self,
        ethr: float = 0.15,
        rthr: float = 0.125,
        bthr: float = 0.01,
        ewin: float = 10000,
        track_stats: bool = False,
    ):
        """
        Initialize the CREST pruner.

        Args:
            ethr (float, optional): Energy threshold. Defaults to ``0.15``.
            rthr (float, optional): RMSD threshold. Defaults to ``0.125``.
            bthr (float, optional): Bond threshold. Defaults to ``0.01``.
            ewin (int, optional): Energy window. Defaults to ``10000``.
            track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
        """
        super().__init__(track_stats)

        self.ethr = ethr
        self.rthr = rthr
        self.bthr = bthr
        self.ewin = ewin

    def is_available(self):
        return crest_available

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

        Returns:
            List[dict]: Updated conformer data.
        """
        if unique_mol_data is None:
            unique_mol_data = []

        all_mol_data = unique_mol_data + current_mol_data
        updated_unique_mol_data, conf_ids = run_cre_check(
            all_mol_data, ethr=self.ethr, rthr=self.rthr, bthr=self.bthr, ewin=self.ewin
        )

        if sort_by_energy:
            updated_unique_mol_data = sorted(
                updated_unique_mol_data, key=lambda x: x["energy"]
            )

        self.n_input_confs = len(all_mol_data)
        self.n_output_confs = len(updated_unique_mol_data)
        self.n_pruned_confs = self.n_input_confs - self.n_output_confs

        if return_ids:
            return updated_unique_mol_data, conf_ids
        else:
            return updated_unique_mol_data
