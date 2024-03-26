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

    def embed_conformers(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        self.mol.EmbedMultipleConfs(n_conformers)
        return self.mol
