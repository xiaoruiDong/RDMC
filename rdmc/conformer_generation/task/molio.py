#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import subprocess
import traceback
from typing import Any, Optional

from rdmc.conformer_generation.task.base import Task


class MolIOTask(Task):
    """
    An abstract class for tasks that deals with molecules and involves I/O operations.

    The mol object that to be operated on, will be the first arguments of run and __call__
    and should has an attribute called `keep_ids` which indicates the indices of the
    conformers to work on in the current task. The number of Trues' in keep_ids will be
    used to determine the number of subtasks; it will be modified by the task to indicate
    the success of the subtasks.

    This class should at least involves a write_input function, an execute function, and a
    read_output function. These will be called by the run function.

    Each subtask will be executed in a separate directory and in a sequential way. The
    parallelized version of this class is under development.
    """

    label = 'MolIOTask'
    # Define the files as {type: name} that are used in the task
    # Usually 'input_file', 'log_file', and 'output_file' are defined
    # (but of course you can change the names)
    files = {'input_file': 'input.in',
             'log_file': 'output.log',
             'output_file': 'output.out',
             'file_type': 'file_name'}
    keep_files = ['*']
    # Define the common directory title for the subtasks
    subtask_dir_name = 'subtask'
    # Whether to create a new mol object for each subtask
    # This is suggested if the task involves changing the
    # molecule geometries
    create_mol_flag = False
    # The attributes to be calculated and attached to the mol object
    # and their initial values
    init_attrs = {}

    @property
    def run_ids(self):
        """
        The indices of the subtasks to be run.

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
        Update the indices of the subtasks to be run.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
        """
        if hasattr(mol, 'keep_ids'):
            self._run_ids = [i for i, keep in enumerate(mol.keep_ids) if keep]
        else:
            self._run_ids = list(range(mol.GetNumConformers()))
        return self.run_ids

    @staticmethod
    def input_writer(mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs):
        """
        The input writer of the task. This function should be implemented by the developer.
        It should return the writer function of the input file. This function should take at
        least two arguments: mol and conf_id, where mol is the molecule to be operated on, and
        conf_id is the index of the conformer to be operated on. However, some tasks may not need
        a input writer, in which case this function can be left as it is.
        """
        raise NotImplementedError

    def write_input_file(self,
                         mol: 'RDKitMol',
                         **kwargs):
        """
        The default input writer of the task. This function iterates through run_ids,
        and call input_writer for each subtask. If failed to write the input file for a
        subtask, the subtask will be marked as failed in mol.keep_ids.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
        """
        for cid in self.run_ids:
            try:
                input_content = self.input_writer(mol=mol,
                                                  conf_id=cid,
                                                  **kwargs)
                with open(self.paths['input_file'][cid], 'w') as f_inp:
                    f_inp.write(input_content)
            except Exception as exc:
                mol.keep_ids[cid] = False
                print(f'Error in writing input file for {self.label}'
                      f' subtask {cid}')
                traceback.print_exc()

        self.update_run_ids(mol=mol)

    def get_execute_command(self,
                            subtask_id: int
                            ) -> list:
        """
        Get the command to execute the subtask. This function should be implemented in the
        child class. This is used by subproc_runner to get the command to execute the subtask.
        It can be left as it is if the child class does not utilize subproc_runner.
        """
        raise NotImplementedError

    def update_n_subtasks(self, mol: 'RDKitMol'):
        """
        Update the number of subtasks, defined by the number of `True`s in the
        keep_ids attribute of the mol object.

        Args:
            mol (RDKitMol): The molecule to be operated on.
        """
        try:
            if not hasattr(mol, 'keep_ids'):
                mol.keep_ids = [bool(conf.GetProp('KeepID'))
                                for conf in mol.GetAllConformers()]
            self.n_subtasks = sum(mol.keep_ids)
        except KeyError:
            # No keepID property, assume all conformers are to be optimized
            self.n_subtasks = mol.GetNumConformers()

    def update_n_success(self):
        """
        Update the number of successful subtasks. This is ran after the task is executed.
        Therefore computed molecule and its properties are stored in self.last_result.
        """
        mol = self.last_result
        self.n_success = sum(mol.keep_ids)
        for conf, keep_id in zip(mol.GetAllConformers(),
                                 mol.keep_ids,):
            conf.SetBoolProp("KeepID", keep_id)
            conf.SetBoolProp(f"{self.label}Success", keep_id)

    def update_paths(self):
        """
        Update the paths of the subtasks. This function creates the subtask directories
        and the paths of the files in the subtask directories according to the subtask_dir_name
        and the files defined in the class.
        """
        subtask_dirs = [osp.join(self.work_dir, f'{self.subtask_dir_name}{subtask_id}')
                        for subtask_id in self.run_ids]
        for subtask_dir in subtask_dirs:
            os.makedirs(subtask_dir, exist_ok=True)
        self.paths = {'subtask_dir': subtask_dirs}
        for ftype, fname in self.files.items():
            self.paths[ftype] = [osp.join(self.paths['subtask_dir'][subtask_id], fname)
                                 for subtask_id in self.run_ids]

    def runner(self,
               subtask_id: int,
               **kwargs):
        """
        The subprocess runner of the task. This function should be implemented in the child class.

        By default, the runner will utilize subprocess module to exectute the command returned by
        get_execute_command.
        """
        with open(self.paths['log_file'][subtask_id], 'w') as f_out:
            subprocess.run(
                self.get_execute_command(subtask_id=subtask_id),
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=self.paths['subtask_dir'][subtask_id],
                check=True,)

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: Any,
                               ):
        """
        Analyze the result of the subtask. This function should be implemented in the child class.
        """
        raise NotImplementedError

    def pre_run(self, mol, **kwargs):
        """
        Pre-run involves the following steps:
        1. Set the number of conformers to be optimized to n_subtasks.
        2. Create the directories and paths for the subtasks.
        """
        # 1. Assign the number of subtasks
        self.update_n_subtasks(mol=mol)

        # 2. Assign the working directory
        self.update_work_dir()

        # 3. Creating all necessary directories and file paths
        # This step will update run_ids (an attribute of the task)
        # to let aware of the subtasks to be run
        # and assign file paths to the paths attribute
        self.update_run_ids(mol=mol)
        self.update_paths()

        # 4. Write the input files
        self.write_input_file(mol=mol, **kwargs)

    @Task.timer
    def run(self, mol, **kwargs):
        """
        Run the task on the molecule.
        """
        # Set up molecule information
        # 1. Create a copy of the molecule if necessary
        # 2. Initialize the attributes of the molecule
        # 3. Update the multiplicity and charge of the molecule
        new_mol = mol.Copy(copy_attrs=['keep_ids']) if self.create_mol_flag else mol
        for attr_name, init_value in self.init_attrs.items():
            setattr(new_mol, attr_name, [init_value] * new_mol.GetNumConformers())
        # update multiplicity and charge
        kwargs['mult'], kwargs['charge'] = self._get_mult_and_chrg(mol=mol,
                                                                   multiplicity=kwargs.get('multiplicity'),
                                                                   charge=kwargs.get('charge'))

        # Run the subtasks in sequence
        for subtask_id in self.run_ids:

            # 1. run the subtask defined in runner
            try:
                subtask_result = self.runner(mol=mol,
                                             subtask_id=subtask_id,
                                             **kwargs)
            except Exception as exc:
                new_mol.keep_ids[subtask_id] = False
                print(f'Error in running subtask {subtask_id} of {self.label}: {exc}')
                traceback.print_exc()
                continue

            # 2. analyze the subtask result
            try:
                self.analyze_subtask_result(mol=new_mol,
                                            subtask_id=subtask_id,
                                            subtask_result=subtask_result,
                                            **kwargs,
                                            )
            except Exception as exc:
                new_mol.keep_ids[subtask_id] = False
                print(f'Error in parsing subtask {subtask_id} of {self.label}: {exc}')
                traceback.print_exc()

        return new_mol

    def post_run(self, **kwargs):
        """
        Set the energy and keepid to conformers.
        """
        # 1. Update the number of successful subtasks
        self.update_n_success()

        # 2. Clean up working directory
        self.clean_work_dir()

    @staticmethod
    def _get_mult_and_chrg(mol: 'RDKitMol',
                           multiplicity: Optional[int],
                           charge: Optional[int]):
        """
        A helper function when parsing multiplicity and charge from the function
        arguments. Use the multiplicity and charge from the molecule if not specified
        """
        if multiplicity is None:
            multiplicity = mol.GetSpinMultiplicity()
        if charge is None:
            charge = mol.GetFormalCharge()
        return multiplicity, charge
