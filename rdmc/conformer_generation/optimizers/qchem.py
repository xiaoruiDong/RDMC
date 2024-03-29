from typing import Optional, Tuple

import numpy as np

from rdmc.conformer_generation.optimizers.base import ConfGenOptimizer
from rdmc.conformer_generation.task.qchem import QChemTask
from rdmc.external.inpwriter import write_qchem_opt


class QChemOptimizer(
    QChemTask,
    ConfGenOptimizer,
):
    """
    The class to optimize TS geometries using the Baker's eigenvector-following (EF) algorithm built in QChem.
    You have to have the QChem package installed to run this optimizer.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                Defaults to ``"wB97x-d3"``.
        basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                Defaults to ``"def2-tzvp"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by QChem. Defaults to ``1``.
        binary_path (str, optional): The path to the QChem binary. Defaults to ``None``, the task will try to locate the binary automatically.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "qchem_opt"
    optimize_ts = False

    def __init__(
        self,
        method: str = "wB97x-d3",
        basis: str = "def2-tzvp",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
        track_stats: bool = False,
    ):
        super(QChemOptimizer, self).__init__(
            method=method,
            basis=basis,
            nprocs=nprocs,
            memory=memory,
            binary_path=binary_path,
        )
        super(QChemTask, self).__init__(track_stats=track_stats)

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

        input_file = work_dir / f"{self.path_prefix}.qcin"
        output_file = work_dir / f"{self.path_prefix}.log"

        # Generate and save the input file
        input_content = write_qchem_opt(
            mol,
            conf_id=conf_id,
            ts=self.optimize_ts,
            method=self.method,
            basis=self.basis,
            mult=multiplicity,
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
                log = self.logparser(output_file)
                if log.success:
                    pos = log.converged_geometries[-1]
                    success = True
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
            except BaseException as e:
                print(f"Got an error when reading the output file ({output_file}): {e}")

        return pos, success, energy, freq
