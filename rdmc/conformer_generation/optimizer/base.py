#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import shutil
from typing import Optional

from rdmc.conformer_generation.task import Task, MolIOTask
from rdmc.conformer_generation.utils import mol_to_sdf


class BaseOptimizer(Task):

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
        for conf, keep_id, e in zip(mol.GetAllConformers(),
                                    mol.keep_ids,
                                    mol.energies):
            conf.SetDoubleProp("Energy", e)
            conf.SetBoolProp("KeepID", keep_id)
            conf.SetBoolProp(f"{self.label}Success", keep_id)
        self.n_success = sum(mol.keep_ids)

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
        a molecule need to be returned as the result, and
        mol.keep_ids (if succeed) and mol.energies should be set in this method.
        """
        raise NotImplementedError

    @staticmethod
    def _get_mult_and_chrg(mol: 'RDKitMol',
                           multiplicity: Optional[int],
                           charge: Optional[int]):
        """
        Use the multiplicity and charge from the molecule if not specified
        """
        if multiplicity is None:
            multiplicity = mol.GetSpinMultiplicity()
        if charge is None:
            charge = mol.GetFormalCharge()
        return multiplicity, charge


class IOOptimizer(MolIOTask):

    def post_run(self, **kwargs):
        """
        Besides setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super().post_run(**kwargs)
        mol = self.last_result
        for conf, energy in zip(mol.GetAllConformers(),
                                mol.energies):
            conf.SetDoubleProp("Energy", energy)

    def save_data(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        mol = self.last_result
        path = osp.join(self.save_dir, "optimized_confs.sdf")

        mol_to_sdf(mol=mol,
                   path=path,)
