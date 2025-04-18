import os
from typing import Optional

from rdmc.conformer_generation.comp_env.orca import orca_available
from rdmc.conformer_generation.comp_env.software import get_binary
from rdmc.conformer_generation.utils import subprocess_runner
from rdmc.external.logparser import ORCALog


class OrcaTask:

    logparser = ORCALog

    def __init__(
        self,
        method: str = "XTB2",
        nprocs: int = 1,
        memory: int = 1,
        binary_path: Optional[str] = None,
    ):
        """
        Initiate the Orca Task.

        Args:
            method (str, optional): The method to be used for the Orca job. you can use the level of theory available in Orca.
                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                Defaults to ``"XTB2"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by Orca. Defaults to ``1``.
            binary_path (str, optional): The path to the Orca binary. Defaults to ``None``, the task will try to locate the binary automatically.
        """
        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.binary_path = binary_path or get_binary("orca")

    def is_available(self):
        """
        Check if the Orca binary is available.

        Returns:
            bool
        """
        return orca_available

    def subprocess_runner(
        self,
        input_path: str,
        output_path: str,
        work_dir: Optional[str] = None,
        command: Optional[list] = None,
    ):
        """
        Run the Orca task with the subprcoess module.

        Args:
            input_path (str): The input file.
            output_path (str): The output file.
            work_dir (str, optional): The working directory. Defaults to ``None``, the current working directory will be used.
            command (list, optional): The command to run. Defaults to ``None``, the command will be ``[self.binary_path, input_path]``.
        """
        # Run the job via subprocess
        if command is None:
            command = [str(self.binary_path), str(input_path)]

        # force orca to be in the PATH and LD_LIBRARY_PATH
        if command[0] != "orca":
            env = os.environ.copy()
            env["PATH"] = os.path.dirname(command[0]) + ":" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = (
                os.path.dirname(command[0]) + ":" + env.get("LD_LIBRARY_PATH", "")
            )

        return subprocess_runner(command, output_path, work_dir, env=env)
