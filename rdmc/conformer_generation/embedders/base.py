from abc import abstractmethod

from rdmc.conformer_generation.task.basetask import BaseTask


class ConfGenEmbedder(BaseTask):
    """
    Base class for conformer generation embedders.
    """

    def __init__(self, track_stats=False):

        super().__init__(track_stats)
        self.iter = 0
        self.n_success = None
        self.percent_success = None
        self.smiles = None

    @abstractmethod
    def run(self, smiles: str, n_conformers: int, **kwargs):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def update_stats(self, exe_time: float, mol, n_conformers: int, *args, **kwargs):
        """
        Update the statistics of the conformer generation.

        Args:
            exe_time (float): Execution time of the conformer generation
            n_conformers (int): Number of conformers planned to generate
        """
        n_success = mol.GetNumConformers()
        stats = {
            "iter": self.iter,
            "time": exe_time,
            "n_success": n_success,
            "percent_success": n_success / n_conformers * 100,
        }
        self.stats.append(stats)

    def __call__(self, *args, **kwargs):

        self.iter += 1
        return super().__call__(*args, **kwargs)
