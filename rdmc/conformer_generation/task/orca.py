from typing import Optional

from rdmc.conformer_generation.comp_env.orca import orca_available
from rdmc.conformer_generation.comp_env.software import get_binary
from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.utils import subprocess_runner
from rdmc.external.logparser import ORCALog


class OrcaTask(BaseTask):

    logparser = ORCALog

    def __init__(
        self,
        method: str = "XTB2",
        nprocs: int = 1,
        memory: int = 1,
        track_stats: bool = False,
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
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
            binary_path (str, optional): The path to the Orca binary. Defaults to ``None``, the task will try to locate the binary automatically.
        """
        super().__init__(track_stats)
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
            command = [self.binary_path, input_path]

        return subprocess_runner(command, output_path, work_dir)
