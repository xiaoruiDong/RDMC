from typing import Optional

from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.comp_env.gaussian import (
    gaussian_available,
    get_default_gaussian_binary,
)


class GaussianTask(BaseTask):

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        track_stats: bool = False,
        binary_path: Optional[str] = None,
    ):
        """
        Initiate the Gaussian optimizer.

        Args:
            method (str, optional): The method to be used for stable species optimization. you can use the level of theory available in Gaussian.
                We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
            binary_path (str, optional): The path to the Gaussian binary. Defaults to ``None``, the task will try to locate the binary automatically,
                by checking if g16, g09, and g03 executable is in the PATH.
        """
        super().__init__(track_stats)
        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.binary_path = binary_path or get_default_gaussian_binary()

    def is_available(self):
        """
        Check if ETKDG embedder is available. Always True under the RDMC Framework

        Returns:
            bool: True
        """
        return gaussian_available
