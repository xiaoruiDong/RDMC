import os
from typing import Optional

import numpy as np

from rdmc.conformer_generation.comp_env.ase import (
    AutoNEB,
    Calculator,
    CalculationFailed,
    QuasiNewton,
)
from rdmc.conformer_generation.comp_env.software import package_available
from rdmc.conformer_generation.comp_env.xtb import xtb_calculator
from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser


class AutoNEBGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the AutoNEB method.

    Args:
        optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        optimizer: "Calculator" = xtb_calculator,
        track_stats: Optional[bool] = False,
    ):
        """
        Initialize the AutoNEB TS initial guesser.

        Args:
            optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)
        self.optimizer = optimizer

    def is_available(self):
        """
        Check if the AutoNEB method is available.
        """
        return package_available["ase"]

    @property
    def attach_calculators(self):
        """
        Set the calculator for each image.
        """

        def fun(images):
            for i in range(len(images)):
                images[i].set_calculator(self.optimizer())

        return fun

    @property
    def optimizer(self):
        """
        Optimizer used by the AutoNEB method.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: "Calculator"):
        if isinstance(optimizer, Calculator):
            self._optimizer = optimizer
        elif hasattr(optimizer, "calculate"):
            # For some exterior calculator, they might be a chance that
            # the optimzier's parent class is not consistent with installed
            # ase Calculator
            # However, in such case, as long as the API is consistent, it
            # is fine.
            self._optimizer = optimizer
        else:
            raise NotImplementedError(
                "The optimizer is not supported, please double check the input optimizer."
            )
        self._optimizer = optimizer

    def run(
        self, mols, multiplicity: Optional[int] = None, save_dir: Optional[str] = None
    ):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """

        ts_guesses, used_rp_combos = {}, []
        for i, (r_mol, p_mol) in enumerate(mols):

            ts_conf_dir = self.work_dir / f"neb_conf{i}"

            r_traj = ts_conf_dir / "ts000.traj"
            p_traj = ts_conf_dir / "ts001.traj"

            r_atoms = r_mol.ToAtoms()
            r_atoms.set_calculator(self.optimizer())
            qn = QuasiNewton(r_atoms, trajectory=r_traj, logfile=None)
            qn.run(fmax=0.05)

            p_atoms = p_mol.ToAtoms()
            p_atoms.set_calculator(self.optimizer())
            qn = QuasiNewton(p_atoms, trajectory=p_traj, logfile=None)
            qn.run(fmax=0.05)

            # need to change dirs bc autoneb path settings are messed up
            cwd = os.getcwd()

            try:
                os.chdir(ts_conf_dir)

                autoneb = AutoNEB(
                    self.attach_calculators,
                    prefix="ts",
                    optimizer="BFGS",
                    n_simul=3,
                    n_max=7,
                    fmax=0.05,
                    k=0.5,
                    parallel=False,
                    maxsteps=[50, 1000],
                )

                autoneb.run()

                ts_guess_idx = np.argmax(autoneb.get_energies())
                ts_guesses[i] = autoneb.all_images[ts_guess_idx].positions

            except (CalculationFailed, AssertionError) as e:
                ts_guesses[i] = None
            finally:
                used_rp_combos.append((r_mol, p_mol))
                os.chdir(cwd)

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [
            ts_mol.GetEditableConformer(i).SetPositions(p)
            for i, p in ts_guesses.items()
            if ts_guesses[i] is not None
        ]

        if self.save_dir:
            self.save_guesses(used_rp_combos, ts_mol)

        ts_mol.KeepIDs = {i: val is not None for i, val in ts_guesses.items()}

        return ts_mol
