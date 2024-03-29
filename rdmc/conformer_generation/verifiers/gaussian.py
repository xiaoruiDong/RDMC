from typing import Optional, Tuple

import numpy as np

from rdmc.conformer_generation.task.gaussian import GaussianTask
from rdmc.conformer_generation.verifiers.base import FreqVerifier
from rdmc.external.inpwriter import write_gaussian_freq


class GaussianFreqVerifier(
    GaussianTask,
    FreqVerifier,
):
    """
    Frequency calculator of Gaussian.

    Args:
        method (str, optional): The method to be used for species optimization. You can use the level of theory available in Gaussian.
            Defaults to ``"GFN2-xTB"``, which is realized by additional scripts provided in the ``rdmc`` package.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        binary_path (str, optional): The path to the Gaussian binary. Defaults to ``None``, the task will try to locate the binary automatically,
            by checking if g16, g09, and g03 executable is in the PATH.
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations.
            Defaults to ``0.0`` cm-1.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "gaussian_freq"

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
        cutoff_frequencies: float = 0.0,
        track_stats: bool = False,
    ):
        super(GaussianFreqVerifier, self).__init__(
            method=method, nprocs=nprocs, memory=memory, binary_path=binary_path
        )
        super(GaussianTask, self).__init__(
            cutoff_frequency=cutoff_frequencies, track_stats=track_stats
        )

    def calc_freq(
        self,
        mol: "RDKitMol",
        conf_id: int,
        multiplicity: int = 1,
        **kwargs,
    ) -> Optional["np.ndarray"]:
        """
        Calculate the frequency of the conformer.

        Args:
            mol (RDKitMol): An RDKitMol object with all guess geometries embedded as conformers.
            conf_id (int): The conformer id.
            multiplicity (int): The multiplicity of the molecule. Defaults to ``1``.

        Returns:
            tuple: pos, success, energy, freq
        """
        work_dir = self.work_dir / f"{self.path_prefix}{conf_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        input_file = work_dir / f"{self.path_prefix}.gjf"
        output_file = work_dir / f"{self.path_prefix}.log"

        # Generate and save the input file
        input_content = write_gaussian_freq(
            mol,
            conf_id=conf_id,
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

        freq = None
        if subprocess_run.returncode != 0:
            print(f"Run into error in conformer optimization.")
        else:
            try:
                log = self.logparser(output_file)
                if log.success:
                    freq = log.freqs

            except Exception as e:
                print(f"Got an error when reading the output file: {e}")

        return freq
