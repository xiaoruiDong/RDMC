#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing initial guess geometries
"""

import json
import os.path as osp

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import mol_to_sdf

class ConformerEmbedder(Task):
    """
    Base class for conformer embedding tasks.
    """

    def pre_run(self,
                *,
                n_conformers: int,
                **kwargs,
                ):
        """
        Set the number of conformers to be generated to n_subtasks.
        """
        self.n_subtasks = n_conformers

    def post_run(self,
                 **kwargs,
                 ):
        """
        Set the number of conformers generated to n_success.
        """
        mol = self.last_result
        for conf, stat in zip(mol.GetAllConformers(),
                              self.status,):
            conf.SetBoolProp("KeepID", stat)

    def save_data(self):
        """
        Save the optimized conformers.
        """
        path = osp.join(self.save_dir, "conformers.sdf")
        mol_to_sdf(mol=self.last_result,
                   path=path)
