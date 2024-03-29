from typing import Optional

from rdmc.conformer_generation.comp_env.gaussian import (
    gaussian_available,
    get_default_gaussian_binary,
)
from rdmc.conformer_generation.utils import subprocess_runner
from rdmc.external.logparser import GaussianLog


class GaussianTask:

    logparser = GaussianLog

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
    ):
        """
        Initiate the Gaussian task.

        Args:
            method (str, optional): The method to be used for the Gaussian task. you can use the level of theory available in Gaussian.
                We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
            binary_path (str, optional): The path to the Gaussian binary. Defaults to ``None``, the task will try to locate the binary automatically,
                by checking if g16, g09, and g03 executable is in the PATH.
        """
        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.binary_path = binary_path or get_default_gaussian_binary()

    def is_available(self):
        """
        Check if the Gaussian binary is available.

        Returns:
            bool
        """
        return gaussian_available

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
            command = [str(self.binary_path), str(input_path)]

        return subprocess_runner(command, output_path, work_dir)
