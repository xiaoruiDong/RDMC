#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing initial guess geometries
"""

import json
import os.path as osp

from rdkit import Chem
from rdmc.conformer_generation.task import Task


class ConformerEmbedder(Task):
    """
    Base class for conformer embedding tasks.
    """

    def pre_run(self,
                n_conformers: int,
                *args,
                **kwargs,
                ):
        """
        Set the number of conformers to be generated to n_subtasks.
        """
        self.n_subtasks = n_conformers

    def post_run(self,
                 *args,
                 **kwargs,
                 ):
        """
        Set the number of conformers generated to n_success.
        """
        self.n_success = self.last_result.GetNumConformers()

    def save_data(self):
        """
        Save the optimized conformers.
        """
        mol = self.last_result

        # Save optimized ts mols
        path = osp.join(self.save_dir, "conformers.sdf")
        try:
            writer = Chem.rdmolfiles.SDWriter(path)
            for i in range(self.n_subtasks):
                writer.write(mol._mol, confId=i)
        except Exception:
            raise
        finally:
            writer.close()
