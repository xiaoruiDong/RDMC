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
        track_stats: Optional[bool] = False,
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

    def run(
        self,
        mols: list,
        multiplicity: Optional[int] = None,
    ):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        # #TODO: May add a support for scratch directory
        # currently use the save directory as the working directory
        # This may not be ideal for some QM software, and whether to add a support
        # for scratch directory is left for future decision

        lot_inp_file = self.work_dir / "qstart.inp"
        lot_inp_str = write_gaussian_gsm(self.method, self.memory, self.nprocs)
        with open(lot_inp_file, "w") as f:
            f.writelines(lot_inp_str)

        ts_guesses, used_rp_combos = {}, []
        for i, (r_mol, p_mol) in enumerate(mols):

            # TODO: Need to clean the logic here, `ts_conf_dir` is used no matter `save_dir` being true
            ts_conf_dir = self.work_dir / f"degsm_conf{i}"
            ts_conf_dir.mkdir(parents=True, exist_ok=True)

            xyz_file = ts_conf_dir / f"degsm_conf{i}.xyz"
            with open(xyz_file, "w") as f:
                f.write(r_mol.ToXYZ())
                f.write(p_mol.ToXYZ())
            used_rp_combos.append((r_mol, p_mol))

            try:
                command = f"{get_binary('gsm')} -xyzfile {xyz_file} -nproc {self.nprocs} -multiplicity {multiplicity} -mode DE_GSM -package Gaussian -lot_inp_file {lot_inp_file} {self.gsm_args}"
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
                    positions = np.array(
                        [line.split()[1:] for line in positions], dtype=float
                    )
                ts_guesses[i] = positions
            except FileNotFoundError:
                ts_guesses[i] = None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [
            ts_mol.GetEditableConformer(i).SetPositions(p)
            for i, p in ts_guesses.items()
            if ts_guesses is not None
        ]

        if self.save_dir:
            self.save_guesses(used_rp_combos, ts_mol)

        ts_mol.KeepIDs = {i: val is not None for i, val in ts_guesses.items()}

        return ts_mol
