#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

import numpy as np

from rdmc.conformer_generation.task import MolTask, MolIOTask
from rdmc.conformer_generation.utils import mol_to_sdf


class BaseOptimizer(MolTask):
    """
    An abstract class for tasks that optimize conformers. Compared to MolTask,
    it saves the optimized conformers (optimized_confs.sdf) to the save_dir.
    """

    def post_run(self, **kwargs):
        """
        Besides setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super().post_run(**kwargs)
        assign_energy_to_conformer(self.last_result)

    def save_data(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        path = osp.join(self.save_dir, "optimized_confs.sdf")
        mol_to_sdf(mol=self.save_dir,
                   path=path,)


class IOOptimizer(MolIOTask):
    """
    An abstract class for tasks that optimize conformers and involve I/O operation.
    Compared to MolTask, it saves the optimized conformers (optimized_confs.sdf) to
    the save_dir. Besides, since some programs also generated Hessian results during
    optimization, this class will also saves the frequencies if available.
    """

    # By default, create a copy of the input molecule in optimization. So, the
    # original molecule will not be modified.
    copy_mol_flag = True
    # The type of calculation. This attribute may be used by the class to write
    # input files or parse output files.
    calc_type = 'opt'
    # The attributes and their initial values to be attached to the mol object before
    # the task. In optimization, the energies and frequencies are expected to be
    # calculated. So, they are included in the init_attrs.
    init_attrs = {'energies': np.nan, 'frequencies': None}

    def post_run(self, **kwargs):
        """
        Besides setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super().post_run(**kwargs)
        assign_energy_to_conformer(self.last_result)

    def save_data(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        path = osp.join(self.save_dir, "optimized_confs.sdf")
        mol_to_sdf(mol=self.last_result,
                   path=path,)


def assign_energy_to_conformer(mol: 'RDKitMol'):
    """
    Assign energy to conformer. This is a helper function assigning energies in
    mol.energies to conformer's Energy property.

    Args:
        mol (RDKitMol): The molecule to be operated on.
    """
    for conf, energy in zip(mol.GetAllConformers(),
                            mol.energies):
        conf.SetDoubleProp("Energy", energy)
