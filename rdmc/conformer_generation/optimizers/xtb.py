from typing import List

from rdmc.conformer_generation.comp_env.xtb.opt import run_xtb_calc
from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.task.xtb import XTBTask
from rdmc.conformer_generation.utils import dict_to_mol


class XTBOptimizer(ConfGenOptimizer, XTBTask):
    """
    Optimizer using the xTB.

    Args:
        method (str, optional): The method to be used for species optimization. Defaults to ``"gff"``.
        level (str, optional): The level of theory. Defaults to ``"normal"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self, method: str = "gff", level: str = "normal", track_stats: bool = False
    ):
        super().__init__(track_stats=track_stats)
        self.method = method
        self.level = level

    def optimize_conformers(
        self,
        mol_data: List[dict],
    ) -> List[dict]:
        """
        Optimize the conformers.

        Args:
            mol_data (List[dict]): The list of conformers to be optimized.

        Returns:
            List[dict]: The list of optimized conformers sorted by energy.
        """
        if len(mol_data) == 0:
            return mol_data

        new_mol = dict_to_mol(mol_data)
        uhf = new_mol.GetSpinMultiplicity() - 1
        correct_atom_mapping = new_mol.GetAtomMapNumbers()

        failed_ids = set()
        all_props = []
        for c_id in range(len(mol_data)):
            try:
                props, opt_mol = run_xtb_calc(
                    new_mol,
                    confId=c_id,
                    job="--opt",
                    return_optmol=True,
                    method=self.method,
                    level=self.level,
                    uhf=uhf,
                )
                all_props.append(props)
            except ValueError as e:
                failed_ids.add(c_id)
                print(e)
                continue

            # opt_mol.SetAtomMapNumbers(correct_atom_mapping)
            # Renumber the molecule based on the atom mapping just set
            # opt_mol.RenumberAtoms()
            positions = opt_mol.GetPositions()
            conf = new_mol.GetEditableConformer(id=c_id)
            conf.SetPositions(positions)
            energy = props["total energy"]
            mol_data[c_id].update(
                {
                    "positions": positions,  # issues if not all opts succeeded?
                    "conf": conf,
                    "energy": energy,
                }
            )

        final_mol_data = [c for i, c in enumerate(mol_data) if i not in failed_ids]

        if self.track_stats:
            self.n_failures = len(failed_ids)
            self.percent_failures = self.n_failures / len(mol_data) * 100
            self.n_opt_cycles = [
                p["n_opt_cycles"] if "n_opt_cycles" in p else -1 for p in all_props
            ]

        return sorted(final_mol_data, key=lambda x: x["energy"])
