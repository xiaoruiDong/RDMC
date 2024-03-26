from rdmc.conformer_generation.embedders.base import ConfGenEmbedder


class ETKDGEmbedder(ConfGenEmbedder):
    """
    Embed conformers using ETKDG.
    """

    _avail = True

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
