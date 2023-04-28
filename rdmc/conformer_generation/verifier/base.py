#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from typing import Optional

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import mol_to_sdf


class BaseVerifier(Task):

    def pre_run(self, mol, **kwargs):
        """
        Set the number of conformers to be optimized to n_subtasks.
        """
        try:
            mol.keep_ids = [bool(conf.GetProp('KeepID'))
                            for conf in mol.GetAllConformers()]
            self.n_subtasks = sum(mol.keep_ids)
        except KeyError:
            # No keepID property, assume all conformers are to be optimized
            self.n_subtasks = mol.GetNumConformers()

    def post_run(self, **kwargs):
        """
        Set the energy and keepid to conformers.
        """
        mol = self.last_result
        for conf, keep_id in zip(mol.GetAllConformers(),
                                 mol.keep_ids,):
            conf.SetBoolProp("KeepID", keep_id)
            conf.SetBoolProp(f"{self.label}Success", keep_id)
        self.n_success = sum(mol.keep_ids)

    def run(self, **kwargs):
        """
        Run the optimization task.
        You need to implement this method in the child class.
        a molecule need to be returned as the result, and
        mol.keep_ids (if succeed) and mol.energies should be set in this method.
        """
        raise NotImplementedError
