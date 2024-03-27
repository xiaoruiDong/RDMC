from typing import Optional

from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.comp_env import qchem_available
from rdmc.conformer_generation.comp_env.software import get_binary


class QChemTask(BaseTask):

    def __init__(
        self,
        method: str = "wB97x-d3",
        basis: str = "def2-tzvp",
        nprocs: int = 1,
        memory: int = 1,
        track_stats: bool = False,
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
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
            binary_path (str, optional): The path to the QChem binary. Defaults to ``None``, the task will try to locate the binary automatically.
        """
        super().__init__(track_stats)
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
