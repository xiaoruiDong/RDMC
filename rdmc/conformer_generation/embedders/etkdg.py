from rdmc import Mol
from rdmc.conformer_generation.embedders.base import ConfGenEmbedder


class ETKDGEmbedder(ConfGenEmbedder):
    """
    Embed conformers using ETKDG.
    """

    def is_available(self):
        """
        Check if ETKDG embedder is available. Always True under the RDMC Framework

        Returns:
            bool: True
        """
        return True

    def run(self, smiles: str, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            smiles (str): SMILES string of the molecule.
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        mol = Mol.FromSmiles(smiles)
        mol.EmbedMultipleConfs(n_conformers)
        return mol
