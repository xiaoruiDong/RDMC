from typing import Optional, Tuple

import numpy as np

from rdmc.conformer_generation.task.qchem import QChemTask
from rdmc.conformer_generation.verifiers.base import FreqVerifier
from rdmc.external.inpwriter import write_qchem_freq


class QChemFreqVerifier(
    QChemTask,
    FreqVerifier,
):
    """
    Frequency calculator of QChem.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                Defaults to ``"wB97x-d3"``.
        basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                Defaults to ``"def2-tzvp"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by QChem. Defaults to ``1``.
        binary_path (str, optional): The path to the QChem binary. Defaults to ``None``, the task will try to locate the binary automatically.
        cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
            imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations.
            Defaults to ``0.0`` cm-1.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "qchem_freq"

    def __init__(
        self,
        method: str = "wB97x-d3",
        basis: str = "def2-tzvp",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
        cutoff_frequencies: float = 0.0,
        track_stats: bool = False,
    ):
        super(QChemFreqVerifier, self).__init__(
            method=method,
            basis=basis,
            nprocs=nprocs,
            memory=memory,
            binary_path=binary_path,
        )
        super(QChemTask, self).__init__(
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
        input_content = write_qchem_freq(
            mol,
            conf_id=conf_id,
            method=self.method,
            basis=self.basis,
            mult=multiplicity,
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
