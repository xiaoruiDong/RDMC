#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing initial guess geometries
"""

import os.path as osp

from rdmc.conformer_generation.task import MolTask
from rdmc.conformer_generation.utils import mol_to_sdf


class ConformerEmbedder(MolTask):
    """
    Base class for conformer embedding tasks.
    """
    def pre_run(self,
                *,
                n_conformers: int,
                **kwargs,
                ):
        """
        Method is called before the run method meant to conduct preprocessing tasks.
        Since at this point, there is no conformer yet, number of subtasks equal to
        the number of conformers to generate.

        Args:
            n_conformers (int): The number of conformer to generate.
        """
        self.n_subtasks = n_conformers
        self._run_ids = list(range(n_conformers))

    def save_data(self):
        """
        Save the optimized conformers to a sdf file.

        For developers: this function is called in __call__ after post_run is called
        and only utilized when `save_dir` attribute is not None.
        """
        path = osp.join(self.save_dir, "conformers.sdf")
        mol_to_sdf(mol=self.last_result,
                   path=path)
