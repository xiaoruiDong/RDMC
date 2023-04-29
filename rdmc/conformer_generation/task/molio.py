#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import subprocess
import traceback
from typing import Any

from rdmc.conformer_generation.task.mol import MolTask


class MolIOTask(MolTask):
    """
    An abstract class for tasks that deals with molecules and involves I/O operations.

    The mol object that to be operated on, will be the first arguments of run and __call__.
    It should already has a few conformers embedded, and an attribute called `keep_ids`
    which indicates the indices of the conformers to work on in the current task.

    In a typical workflow, the task will create the working directories and necessary
    input files in the pre_run process; preprocessing the molecule input, execute the actual
    task by subprocesses, and parse the output files in the run; and update the job status
    information and copy important results to saving directory and remove the working directory.

    For developers:
    This class should at minimum involves a `files` attribute, which is a dictionary of the
    file type: file name used/generated/analyzed in the task; a `subtask_dir_name` attribute,
    which is the common directory title for the subtasks; a `get_execute_command` to obtain
    the command to execute the task; and a `analyze_subtask_result` function to analyze the
    result of each subtask and assign the result (e.g., energy) to the mol object.

    Besides, you may also want to define a `input_writer` function that returns the input file
    content to allow writing input files for each subtask. You can also re-define the `runner`
    and `run` function in case you don't want to use subprocesses to execute the task, and
    and execute each subtask in serial.
    """

    # Define the files as {type: name} that are used in the task
    # Usually 'input_file', 'log_file', and 'output_file' are defined
    # (but of course you can change the names)
    files = {'input_file': 'input.in',
             'log_file': 'output.log',
             'output_file': 'output.out',
             'file_type': 'file_name'}
    # Define the files to be kept after the task is finished; usually the output/log files
    # are kept, but you can change it to any file names. To keep all files, set it to ['*']
    keep_files = ['*']
    # Define the common directory title for the subtasks
    subtask_dir_name = 'subtask'

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

        # update run_ids so job failed in input generation won't run in later steps
        self.update_run_ids(mol=mol)

    def get_execute_command(self,
                            subtask_id: int,
                            ) -> list:
        """
        Get the command to execute the subtask. This function should be implemented in the
        child class. This is used by subproc_runner to get the command to execute the subtask.
        It can be left as it is if the child class does not utilize subproc_runner.
        """
        raise NotImplementedError

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
        The subprocess runner of the task. This function might be re-implemented in the child class.

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
                               **kwargs,
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
        3. Write the input files for the subtasks.
        """
        # 1. Assign the number of subtasks
        self.update_n_subtasks(mol=mol)

        # 2. Assign the working directory
        self.update_work_dir()

        # 3. Creating all necessary directories and file paths
        # This step will update run_ids (an attribute of the task)
        # to let aware of the subtasks to be run
        # and assign file paths to the paths attribute
        self.update_paths()

        # 4. Write the input files
        self.write_input_file(mol=mol, **kwargs)

    @MolTask.timer
    def run(self, mol, **kwargs):
        """
        Run the task on the molecule.
        """
        # Set up molecule information
        # 1. Create a copy of the molecule if necessary
        # 2. Initialize the attributes of the molecule
        # 3. Update the multiplicity and charge of the molecule
        new_mol = self.update_mol(mol=mol)
        # update multiplicity and charge
        kwargs['mult'], kwargs['charge'] = \
                self._get_mult_and_chrg(mol=mol,
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
