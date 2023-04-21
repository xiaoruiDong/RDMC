#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.embedder.conformer import ConformerEmbedder


class ETKDGEmbedder(ConformerEmbedder):
    """
    Embed conformers using the ETKDG method.
    """
    @ConformerEmbedder.timer
    def run(self,
            mol: 'RDKitMol',
            n_conformers: int,
            **kwargs,):
        try:
            mol.EmbedMultipleConfs(n_conformers)
            mol.keep_ids = [True] * n_conformers
        except Exception:
            mol.keep_ids = [False] * n_conformers
            # Todo: log the error
        return mol


class RandomEmbedder(ConformerEmbedder):
    """
    Embed conformers using the random coordinates.
    """
    @ConformerEmbedder.timer
    def run(self,
            mol: 'RDKitMol',
            n_conformers: int,
            **kwargs,):
        mol.EmbedMultipleNullConfs(n_conformers,
                                   random=True)
        mol.keep_ids = [True] * n_conformers  # This method shouldn't fail
        return mol
