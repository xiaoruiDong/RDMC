#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import mol_to_sdf


class BaseOptimizer(Task):

    def pre_run(self, mol, **kwargs):
        """
        Set the number of conformers to be optimized to n_subtasks.
        """
        try:
            self.status = [bool(conf.GetProp('KeepID'))
                           for conf in mol.GetAllConformers()]
            self.n_subtasks = sum(self.status)
        except KeyError:
            # No keepID property, assume all conformers are to be optimized
            self.n_subtasks = mol.GetNumConformers()

    def post_run(self, **kwargs):
        """
        Set the energy and status to conformers.
        """
        mol = self.last_result
        for conf, stat, e in zip(mol.GetAllConformers(),
                                 self.status,
                                 self.energies):
            conf.SetDoubleProp("Energy", e)
            conf.SetBoolProp("KeepID", stat)

    def save_data(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        mol = self.last_result
        path = osp.join(self.save_dir, "optimized_confs.sdf")

        mol_to_sdf(mol=mol,
                   path=path,)

    def run(self, **kwargs):
        """
        Run the optimization task.
        You need to implement this method in the child class.
        self.status (if succeed) and self.energies should be set in this method.
        a molecule need to be returned as the result.
        """
        raise NotImplementedError