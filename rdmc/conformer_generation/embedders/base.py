from time import time

from rdmc import RDKitMol
from rdmc.conformer_generation.utils import mol_to_dict
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

    def update_mol(self, smiles: str):
        """
        Update the molecule graph based on the SMILES string.

        Args:
            smiles (str): SMILES string of the molecule
        """
        # Only update the molecule if smiles is changed
        # Only copy the molecule graph from the previous run rather than conformers
        if smiles != self.smiles:
            self.smiles = smiles
            self.mol = RDKitMol.FromSmiles(smiles)
        else:
            # Copy the graph but remove conformers
            self.mol = self.mol.Copy(quickCopy=True)

    def embed_conformers(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def update_stats(self, n_trials: int, time: float = 0.0) -> dict:
        """
        Update the statistics of the conformer generation.

        Args:
            n_trials (int): Number of trials
            time (float, optional): Time spent on conformer generation. Defaults to ``0.``.

        Returns:
            dict: Statistics of the conformer generation
        """
        n_success = self.mol.GetNumConformers()
        self.n_success = n_success
        self.percent_success = n_success / n_trials * 100
        stats = {
            "iter": self.iter,
            "time": time,
            "n_success": self.n_success,
            "percent_success": self.percent_success,
        }
        self.stats.append(stats)
        return stats

    def write_mol_data(self):
        """
        Write the molecule data.

        Returns:
            dict: Molecule data.
        """
        return mol_to_dict(self.mol, copy=False, iter=self.iter)

    def __call__(self, smiles: str, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            smiles (str): SMILES string of the molecule.
            n_conformers (int): Number of conformers to generate.

        Returns:
            dict: Molecule data.
        """
        self.iter += 1
        time_start = time()
        self.update_mol(smiles)
        self.embed_conformers(n_conformers)
        mol_data = self.write_mol_data()

        if not self.track_stats:
            return mol_data

        time_end = time()
        self.update_stats(n_trials=n_conformers, time=time_end - time_start)
        return mol_data
