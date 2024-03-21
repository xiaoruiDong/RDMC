import os
import numpy as np
from typing import Optional

# Use ASE for the AutoNEB method
_ase_avail = True
try:
    from ase import Atoms
    from ase.autoneb import AutoNEB
    from ase.calculators.calculator import CalculationFailed, Calculator
    from ase.optimize import QuasiNewton
except BaseException:
    _ase_avail = False
    print("No ASE installation detected. Skipping import...")

# Use xtb ase calculator defined in the xtb-python module
try:
    from xtb.ase.calculator import XTB
except BaseException:
    XTB = "xtb-python not installed"  # Defined to provide informative error message.
    print(
        "XTB cannot be used for AutoNEBGuesser as its ASE interface imported incorrectly. Skipping import..."
    )


class AutoNEBGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the AutoNEB method.

    Args:
        optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    _avail = _ase_avail

    def __init__(
        self, optimizer: "Calculator" = XTB, track_stats: Optional[bool] = False
    ):
        """
        Initialize the AutoNEB TS initial guesser.

        Args:
            optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(AutoNEBGuesser, self).__init__(track_stats)
        self.optimizer = optimizer

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
        try:
            assert isinstance(
                optimizer, Calculator
            ), f"Invalid optimizer used ('{optimizer}'). Please use ASE calculators."
        except NameError:
            print(
                "ASE.Calculator was not correctly imported, thus AutoNEBGuesser can not be used."
            )
        self._optimizer = optimizer

    def generate_ts_guesses(
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

        ts_guesses, used_rp_combos = [], []
        for i, (r_mol, p_mol) in enumerate(mols):

            # TODO: Need to clean the logic here, `ts_conf_dir` is used no matter `save_dir` being true
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"neb_conf{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            r_traj = os.path.join(ts_conf_dir, "ts000.traj")
            p_traj = os.path.join(ts_conf_dir, "ts001.traj")

            r_coords = r_mol.GetConformer().GetPositions()
            r_numbers = r_mol.GetAtomicNumbers()
            r_atoms = Atoms(positions=r_coords, numbers=r_numbers)
            r_atoms.set_calculator(self.optimizer())
            qn = QuasiNewton(r_atoms, trajectory=r_traj, logfile=None)
            qn.run(fmax=0.05)

            p_coords = p_mol.GetConformer().GetPositions()
            p_numbers = p_mol.GetAtomicNumbers()
            p_atoms = Atoms(positions=p_coords, numbers=p_numbers)
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
                os.chdir(cwd)

                used_rp_combos.append((r_mol, p_mol))
                ts_guess_idx = np.argmax(autoneb.get_energies())
                ts_guesses.append(autoneb.all_images[ts_guess_idx].positions)

            except (CalculationFailed, AssertionError) as e:
                os.chdir(cwd)

        if len(ts_guesses) == 0:
            return None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [
            ts_mol.GetEditableConformer(i).SetPositions(p)
            for i, p in enumerate(ts_guesses)
        ]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol)

        return ts_mol
