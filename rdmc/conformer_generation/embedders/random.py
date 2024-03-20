from rdmc.conformer_generation.embedders.base import ConfGenEmbedder


class RandomEmbedder(ConfGenEmbedder):
    """
    Embed conformers with coordinates of random numbers.
    """

    def embed_conformers(self, n_conformers: int):
        """
        Embed conformers according to the molecule graph.

        Args:
            n_conformers (int): Number of conformers to generate.

        Returns:
            mol: Molecule with conformers.
        """
        self.mol.EmbedMultipleNullConfs(n_conformers, random=True)
        return self.mol
