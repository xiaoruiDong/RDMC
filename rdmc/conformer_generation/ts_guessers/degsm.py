import subprocess
from typing import Optional

import numpy as np


from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser
from rdmc.external.inpwriter import write_gaussian_gsm
from rdmc.conformer_generation.comp_env import gsm_available
from rdmc.conformer_generation.comp_env.software import get_binary


class DEGSMGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the DE-GSM method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        gsm_args: Optional[str] = "",
        track_stats: bool = False,
    ):
        """
        Initialize the DE-GSM TS initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super().__init__(track_stats)
        self.gsm_args = gsm_args
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

    def is_available(self) -> bool:
        """
        Check if the DE-GSM method is available.

        Returns:
            bool: ``True`` if the DE-GSM method is available, ``False`` otherwise.
        """
        return gsm_available

    def generate_ts_guess(
        self, rmol, pmol, multiplicity: Optional[int] = None, conf_id: int = 0, **kwargs
    ):
        """
        Generate a single TS guess.

        Args:
            rmol (RDKitMol): The reactant molecule in RDKitMol with 3D conformer saved with the molecule.
            pmol (RDKitMol): The product molecule in RDKitMol with 3D conformer saved with the molecule.
            multiplicity(int, optional): The multiplicity of the molecule. Defaults to 1.
            conf_id (int, optional): The id of the TS guess job.

        Returns:
            Tuple[np.ndarray, bool]: The generated guess positions and the success status.
        """
        ts_conf_dir = self.work_dir / f"degsm_conf{conf_id}"
        ts_conf_dir.mkdir(parents=True, exist_ok=True)

        lot_inp_file = ts_conf_dir / "qstart.inp"
        lot_inp_str = write_gaussian_gsm(self.method, self.memory, self.nprocs)
        with open(lot_inp_file, "w") as f:
            f.writelines(lot_inp_str)

        xyz_file = ts_conf_dir / f"degsm_conf{conf_id}.xyz"
        rxyz, pxyz = rmol.ToXYZ(), pmol.ToXYZ()
        with open(xyz_file, "w") as f:
            f.write(rxyz)
            f.write(pxyz)

        multiplicity = multiplicity if multiplicity is not None else 1

        command = (
            f"{get_binary('gsm')} "
            f"-xyzfile {xyz_file} "
            f"-nproc {self.nprocs} "
            f"-multiplicity {multiplicity} "
            f"-mode DE_GSM "
            f"-package Gaussian "
            f"-lot_inp_file {lot_inp_file} "
            f"{self.gsm_args}"
        )

        try:
            with open(ts_conf_dir / "degsm.log", "w") as f:
                subprocess.run(
                    [command],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=ts_conf_dir,
                    shell=True,
                )

            tsnode_path = ts_conf_dir / "TSnode_0.xyz"
            with open(tsnode_path) as f:
                positions = f.read().splitlines()[2:]
            pos = np.array([line.split()[1:] for line in positions], dtype=float)
            success = True

        except FileNotFoundError:
            pos, success = None, False

        return pos, success
