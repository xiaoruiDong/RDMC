from typing import Optional

from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.comp_env.orca import orca_available
from rdmc.conformer_generation.comp_env.software import get_binary


class OrcaTask(BaseTask):

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
