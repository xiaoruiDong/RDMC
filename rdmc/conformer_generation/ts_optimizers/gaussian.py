from typing import Optional, Tuple

import numpy as np

from rdmc.conformer_generation.task.gaussian import GaussianTask
from rdmc.conformer_generation.ts_optimizers.base import TSOptimizer
from rdmc.external.inpwriter import write_gaussian_opt


class GaussianOptimizer(
    TSOptimizer,
    GaussianTask,
):
    """
    The class to optimize TS geometries using the Berny algorithm built in Gaussian.
    You have to have the Gaussian package installed to run this optimizer

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do.
                                Defaults to ``"GFN2-xTB"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "gaussian_opt"

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
        input_content = write_gaussian_opt(
            mol,
            conf_id=conf_id,
            ts=True,
            method=self.method,
            mult=multiplicity,
            nprocs=self.nprocs,
            memory=self.memory,
        )

        with open(input_file, "w") as f:
            f.writelines(input_content)

        subprocess_run = self.subprocess_runner(
            input_file,
            output_file,
            work_dir,
        )

        if subprocess_run.returncode != 0:
            print(f"Run into error in TS optimization.")
        else:
            try:
                log = self.logparser(output_file)
                if log.success:
                    pos = log.converged_geometries[-1]
                    success = True
                    energy = log.get_scf_energies(relative=False)[-1]
                    freq = log.freqs
                else:
                    pos = log.all_geometries[-1]
                    energy = log.get_scf_energies(relative=False, converged=False)[-1]
            except BaseException as e:
                print(f"Got an error when reading the output file ({output_file}): {e}")

        return pos, success, energy, freq
