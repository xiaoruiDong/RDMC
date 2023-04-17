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
        mol.EmbedMultipleConfs(n_conformers)
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
        return mol
