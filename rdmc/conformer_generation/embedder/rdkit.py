#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.embedder.conformer import ConformerEmbedder
from rdmc.conformer_generation.utils import timer


class ETKDGEmbedder(ConformerEmbedder):
    """
    Embed conformers using the ETKDG method.
    """
    @timer
    def run(self,
            mol: 'RDKitMol',
            n_conformers: int,
            **kwargs,):
        try:
            mol.EmbedMultipleConfs(n_conformers)
        except Exception as exc:
            self.keep_ids = [False] * n_conformers
            # Todo: log the error
        else:
            self.keep_ids = [True] * n_conformers
        return mol


class RandomEmbedder(ConformerEmbedder):
    """
    Embed conformers using the random coordinates.
    """
    @timer
    def run(self,
            mol: 'RDKitMol',
            n_conformers: int,
            **kwargs,):
        mol.EmbedMultipleNullConfs(n_conformers,
                                   random=True)
        self.keep_ids = [True] * n_conformers  # This method shouldn't fail
        return mol
