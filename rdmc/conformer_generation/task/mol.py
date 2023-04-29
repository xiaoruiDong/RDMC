#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from rdmc.conformer_generation.task.base import Task


class MolTask(Task):
    """
    An abstract class for tasks that deals with molecules and conformers.

    The mol object that to be operated on, will be the first arguments of run and __call__.
    It should already has a few conformers embedded, and an attribute called `keep_ids`
    which indicates the indices of the conformers to work on in the current task.
    """
    # In some cases, it makes sense to not overwrite the original molecule but to
    # create a copy of the molecule (e.g., in optimization, all conformer coordinates
    # are expected to change after the task, therefore it may be better to keep the old
    # mol, so that initial coordinates are not lost). Set the copy_mol_flag to `True` to
    # create a copy of the molecule. Defaults to False.
    copy_mol_flag = False
    # The attributes and their initial values to be attached to the mol object before
    # the task. This avoid attribute errors when assigning results to the molecule in
    # the later steps. It is suggested to include properties to be calculated during
    # the task.
    init_attrs = {}

    @property
    def run_ids(self):
        """
        The indices of the subtasks (conformers) to be run. This is the attribute
        used in the pre_run and run method to monitor which conformer
        should be operated on.

        Returns:
            list: The indices of the subtasks to be run.
        """
        try:
            return self._run_ids
        except AttributeError:
            return []

    def update_run_ids(self,
                       mol: 'RDKitMol'):
        """
        Update the indices of the subtasks to be run. This method uses the keep_ids
        (a list of True or False) of the given molecule to determine which conformers
        should be operated on. If the molecule does not have keep_ids, all conformers
        will be operated on. The result will be stored in self.run_ids and returned.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.

        Returns:
            list: The indices of the subtasks to be run (the same as self.run_ids)
        """
        if hasattr(mol, 'keep_ids'):
            self._run_ids = [i for i, keep in enumerate(mol.keep_ids) if keep]
        elif mol.GetNumConformers() > 0 and mol.GetConformer().HasProp('KeepID'):
            # Conformers have KeepID property, but for some reason has not assigned
            # to the molecule.
            mol.keep_ids = [bool(conf.GetProp('KeepID'))
                            for conf in mol.GetAllConformers()]
            self._run_ids = [i for i, keep in enumerate(mol.keep_ids) if keep]
        else:
            # Run all conformers
            self._run_ids = list(range(mol.GetNumConformers()))
        return self.run_ids

    def update_n_subtasks(self, mol: 'RDKitMol'):
        """
        Update the number of subtasks, defined by the number of `True`s in the
        keep_ids attribute of the mol object.

        For developers: This method is designed to run in the beginning of the
        pre_run method and should be only called once.

        Args:
            mol (RDKitMol): The molecule to be operated on.
        """
        self.n_subtasks = len(self.update_run_ids(mol))

    def update_n_success(self):
        """
        Update the number of successful subtasks, defined by the number of `True`s
        in the keep_ids attribute of the mol that stored in self.last_result.
        It will also update the KeepID and XXXSuccess properties of the conformers
        based on the keep_ids attribute (Set both properties to True if keep_ids is
        True, otherwise False).

        For developers: this method is designed to run in the post_run method and
        should be only called once.
        """
        mol = self.last_result
        self.n_success = sum(mol.keep_ids)
        for conf, keep_id in zip(mol.GetAllConformers(),
                                 mol.keep_ids,):
            conf.SetBoolProp("KeepID", keep_id)
            conf.SetBoolProp(f"{self.label}Success", keep_id)

    def update_mol(self,
                   mol: 'RDKitMol',
                   ) -> 'RDKitMol':
        """
        # Set up molecule information.
        # 1. Create a copy of the molecule if necessary
        # 2. Initialize the attributes of the molecule
        """
        new_mol = mol.Copy(copy_attrs=['keep_ids']) if self.copy_mol_flag else mol
        for attr_name, init_value in self.init_attrs.items():
            setattr(new_mol, attr_name, [init_value] * new_mol.GetNumConformers())
        return new_mol

    def pre_run(self,
                mol: 'RDKitMol',
                **kwargs):
        """
        Method is called before the run method meant to conduct preprocessing tasks.
        In this basic implementation, it will just update the number of subtasks.

        For developers: **kwargs is kept as pre_run, run, and post_run use the same input
        arguments. This allows accepting irrelevant arguments without raising errors.

        Args:
            mol ('RDKitMol'):  The molecule with conformers to be operated on.
        """
        self.update_n_subtasks(mol=mol)

    @Task.timer
    def run(self,
            mol: 'RDKitMol',
            **kwargs):
        """
        Run the task on the molecule.
        All child classes should implement this method and has at least `mol`
        as the argument.

        Args:
            mol (RDKitMol): RDKitMol object
        """
        raise NotImplementedError

    def post_run(self, **kwargs):
        """
        Method is called before the run method meant to conduct preprocessing tasks.
        In this basic implementation, it will just update the number of subtasks.

        For developers: **kwargs is kept as pre_run, run, and post_run use the same input
        arguments. This allows accepting irrelevant arguments without raising errors.

        Args:
            mol ('RDKitMol'):  The molecule with conformers to be operated on.
        """
        # Update the number of successful subtasks
        self.update_n_success()

    @staticmethod
    def _get_mult_and_chrg(mol: 'RDKitMol',
                           multiplicity: Optional[int],
                           charge: Optional[int]):
        """
        A helper function when parsing multiplicity and charge from the function
        arguments. Use the multiplicity and charge from the molecule if not specified.
        """
        if multiplicity is None:
            multiplicity = mol.GetSpinMultiplicity()
        if charge is None:
            charge = mol.GetFormalCharge()
        return multiplicity, charge
