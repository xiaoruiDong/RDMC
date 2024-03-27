from typing import Optional

import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.task.orca import OrcaTask
from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
from rdmc.external.inpwriter import write_orca_opt
from rdtools.conversion.xyz import xyz_to_coords


class OrcaOptimizer(TSOptimizer, OrcaTask):
    """
    The class to optimize TS geometries using the Berny algorithm built in Orca.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                Defaults to ``"XTB2"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "orca_opt"

    def __init__(self, **kwargs):
        super(TSOptimizer, self).__init__(**kwargs)

    def run_opt(
        self,
        mol: "RDKitMol",
        conf_id: int,
        multiplicity: int = 1,
        **kwargs,
    ) -> Tuple[Optional["np.ndarray"], bool, float, Optional["np.ndarray"]]:
        """
        Optimize the TS guesses.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            conf_id (int): The conformer id.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.

        Returns:
            tuple: pos, success, energy, freq
        """
        pos, success, energy, freq = None, False, np.nan, None

        work_dir = self.work_dir / f"{self.path_prefix}{conf_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        input_file = work_dir / f"{self.path_prefix}.gjf"
        output_file = work_dir / f"{self.path_prefix}.log"

        # Generate and save the input file
        input_content = write_orca_opt(
            mol,
            conf_id=conf_id,
            ts=True,
            method=self.method,
            mult=multiplicity,
            nprocs=self.nprocs,
        )

        with open(input_file, "w") as f:
            f.writelines(input_content)

        # Run the job via subprocess
        subprocess_run = self.subprocess_runner(
            input_file,
            output_file,
            work_dir,
        )

        if subprocess_run.returncode != 0:
            print(f"Run into error in TS optimization.")
        else:
            try:
                with open(work_dir / f"{self.path_prefix}.xyz", "r") as f:
                    pos = xyz_to_coords(f.read(), header=True)
                freq = self.extract_frequencies(
                    output_file,
                    mol.GetNumAtoms(),
                )
                try:
                    energy = self.logparser(output_file).get_scf_energies(
                        relative=False
                    )[-1]
                except BaseException as e:
                    pass
                success = True
            except BaseException as e:
                print(f"Got an error when reading the output file ({output_file}): {e}")

        return pos, success, energy, freq

    @staticmethod
    def extract_frequencies(log_path: str, n_atoms: int):
        """
        Extract frequencies from the Orca opt job.

        Args:
            save_dir (str): Path where Orca logs are saved.
            n_atoms (int): The number of atoms in the molecule.

        Returns:
            np.ndarray: The frequencies in cm-1.
        """

        with open(log_path, "r") as f:
            orca_data = f.readlines()

        dof = 3 * n_atoms
        orca_data.reverse()
        freq_idx = None
        for i, line in enumerate(orca_data):
            if "VIBRATIONAL FREQUENCIES" in line:
                freq_idx = i
                break
        if freq_idx:
            freqs = orca_data[freq_idx - 4 - dof : freq_idx - 4]
            freqs.reverse()
            return np.array([float(line.split()[1]) for line in freqs])
        else:
            return None
