from rdmc.conformer_generation.embedders.base import ConfGenEmbedder


class RandomEmbedder(ConfGenEmbedder):
    """
    Embed conformers with coordinates of random numbers.
    """

    def is_available(self):
        """
        Check if Random embedder is available. Always True under the RDMC Framework.

        Returns:
            bool: True
        """
        return True

    def run(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        self.mol.EmbedMultipleNullConfs(n_conformers, random=True)
        return self.mol
