from typing import Optional

from rdmc.conformer_generation.comp_env import qchem_available
from rdmc.conformer_generation.comp_env.software import get_binary
from rdmc.conformer_generation.utils import subprocess_runner
from rdmc.external.logparser import QChemLog


class QChemTask:

    logparser = QChemLog

    def __init__(
        self,
        method: str = "wB97x-d3",
        basis: str = "def2-tzvp",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
    ):
        """
        Initiate the QChem Task.

        Args:
            method (str, optional): The method to be used for the QChem task. you can use the method available in QChem.
                Defaults to ``"wB97x-d3"``.
            basis (str, optional): The method to be used for the QChem task. you can use the basis available in QChem.
                Defaults to ``"def2-tzvp"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by QChem. Defaults to ``1``.
            binary_path (str, optional): The path to the QChem binary. Defaults to ``None``, the task will try to locate the binary automatically.
        """
        self.method = method
        self.basis = basis
        self.nprocs = nprocs
        self.memory = memory
        self.binary_path = binary_path or get_binary("qchem")

    def is_available(self):
        """
        Check if the QChem binary is available.

        Returns:
            bool
        """
        return qchem_available

    def subprocess_runner(
        self,
        input_path: str,
        output_path: str,
        work_dir: Optional[str] = None,
        command: Optional[list] = None,
    ):
        """
        Run the Gaussian task with the subprcoess module.

        Args:
            input_path (str): The input file.
            output_path (str): The output file.
            work_dir (str, optional): The working directory. Defaults to ``None``, the current working directory will be used.
            command (list, optional): The command to run. Defaults to ``None``, the command will be ``[self.binary_path, input_path]``.
        """
        # Run the job via subprocess
        if command is None:
            command = [self.binary_path, "-nt", str(self.nprocs), input_path]

        return subprocess_runner(command, output_path, work_dir)
