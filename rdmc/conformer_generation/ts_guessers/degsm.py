import os
import subprocess
import tempfile
from typing import Optional

import numpy as np


from rdmc.conformer_generation.ts_guessers.base import TSInitialGuesser
from rdmc.external.inpwriter import write_gaussian_gsm
from rdmc.conformer_generation.comp_env import gsm_available


class DEGSMGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the DE-GSM method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    _avail = gsm_available

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
        super(DEGSMGuesser, self).__init__(track_stats)
        self.gsm_args = gsm_args
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

        try:
            self.gsm_entry_point = os.environ["gsm"]
        except KeyError:
            raise RuntimeError("No GSM entry point is found in the PATH.")

    def generate_ts_guesses(
        self,
        mols: list,
        multiplicity: Optional[int] = None,
        save_dir: Optional[str] = None,
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
        # #TODO: May add a support for scratch directory
        # currently use the save directory as the working directory
        # This may not be ideal for some QM software, and whether to add a support
        # for scratch directory is left for future decision
        work_dir = os.path.abspath(save_dir) if save_dir else tempfile.mkdtemp()
        lot_inp_file = os.path.join(work_dir, "qstart.inp")
        lot_inp_str = write_gaussian_gsm(self.method, self.memory, self.nprocs)
        with open(lot_inp_file, "w") as f:
            f.writelines(lot_inp_str)

        ts_guesses, used_rp_combos = [], []
        for i, (r_mol, p_mol) in enumerate(mols):

            # TODO: Need to clean the logic here, `ts_conf_dir` is used no matter `save_dir` being true
            ts_conf_dir = os.path.join(work_dir, f"degsm_conf{i}")
            if not os.path.exists(ts_conf_dir):
                os.makedirs(ts_conf_dir)

            r_xyz = r_mol.ToXYZ()
            p_xyz = p_mol.ToXYZ()

            xyz_file = os.path.join(ts_conf_dir, f"degsm_conf{i}.xyz")
            with open(xyz_file, "w") as f:
                f.write(r_xyz)
                f.write(p_xyz)

            try:
                command = f"{self.gsm_entry_point} -xyzfile {xyz_file} -nproc {self.nprocs} -multiplicity {multiplicity} -mode DE_GSM -package Gaussian -lot_inp_file {lot_inp_file} {self.gsm_args}"
                with open(os.path.join(ts_conf_dir, "degsm.log"), "w") as f:
                    gsm_run = subprocess.run(
                        [command],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=ts_conf_dir,
                        shell=True,
                    )
                used_rp_combos.append((r_mol, p_mol))
                tsnode_path = os.path.join(ts_conf_dir, "TSnode_0.xyz")
                with open(tsnode_path) as f:
                    positions = f.read().splitlines()[2:]
                    positions = np.array(
                        [line.split()[1:] for line in positions], dtype=float
                    )
                ts_guesses.append(positions)
            except FileNotFoundError:
                pass

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
