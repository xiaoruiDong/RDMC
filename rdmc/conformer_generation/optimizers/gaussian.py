from typing import Optional, Tuple

import numpy as np

from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.task.gaussian import GaussianTask
from rdmc.external.inpwriter import write_gaussian_opt


class GaussianOptimizer(
    GaussianTask,
    ConfGenOptimizer,
):
    """
    Optimizer using the Gaussian.

    Args:
        method (str, optional): The method to be used for species optimization. You can use the level of theory available in Gaussian.
            Defaults to ``"GFN2-xTB"``, which is realized by additional scripts provided in the ``rdmc`` package.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        binary_path (str, optional): The path to the Gaussian binary. Defaults to ``None``, the task will try to locate the binary automatically,
            by checking if g16, g09, and g03 executable is in the PATH.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "gaussian_opt"
    optimize_ts = False

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
        track_stats: bool = False,
    ):
        super(GaussianOptimizer, self).__init__(
            method=method, nprocs=nprocs, memory=memory, binary_path=binary_path
        )
        super(GaussianTask, self).__init__(track_stats=track_stats)

    def run_opt(
        self,
        mol: "RDKitMol",
        conf_id: int,
        multiplicity: int = 1,
        **kwargs,
    ) -> Tuple[Optional["np.ndarray"], bool, float, Optional["np.ndarray"]]:
        """
        Optimize the conformer.

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
            ts=self.optimize_ts,
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
            print(f"Run into error in conformer optimization.")
        else:
            try:
                log = self.logparser(output_file)
                if log.success:
                    pos = log.converged_geometries[-1]
                    energy = log.get_scf_energies(relative=False)[-1]
                    freq = log.freqs

                    # Check connectivity
                    # difference 2 compared to TSGaussianOptimizer
                    if not self.optimize_ts:
                        post_adj_mat = log.get_mol(
                            refid=log.num_all_geoms - 1,  # The last geometry in the job
                            converged=False,
                            sanitize=False,
                            backend="openbabel",
                        ).GetAdjacencyMatrix()
                        pre_adj_mat = mol.GetAdjacencyMatrix()
                        success = np.array_equal(pre_adj_mat, post_adj_mat)
                    else:
                        success = True

                else:
                    pos = log.all_geometries[-1]
                    energy = log.get_scf_energies(relative=False, converged=False)[-1]

            except Exception as e:
                print(f"Got an error when reading the output file: {e}")

        return pos, success, energy, freq
