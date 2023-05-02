#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import subprocess
import traceback
from typing import Any, List

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
    # Define the common directory title for the subtasks
    subtask_dir_name = 'subtask'
    # While for many tasks, each subtask has its own directory and runs separately,
    # some tasks may only need to create one directory for all subtasks, and only need
    # to run once. In this case, set the singleshot_subtask to True.
    singleshot_subtask = False

    @staticmethod
    def input_writer(mol: 'RDKitMol',
                     conf_id: int,
                     **kwargs,
                     ) -> str:
        """
        The input writer of the task. The input geometry is based on the conf_id-th conformer of
        the mol object. It also takes the kwargs as input, which is useful for tasks that need
        additional input information.

        For developers: This function should be implemented by the developer in the child class.
        It should return the writer function of the input file. This function should take at least
        two arguments: mol and conf_id, where mol is the molecule to be operated on, and
        conf_id is the index of the conformer to be operated on. However, some tasks may not need
        a input writer, in which case this function can be left as it is.
        E.g., for the simple case where only xyz coordinates are needed, the input writer can be
        defined as:
        ```
        @staticmethod
        def input_writer(mol: 'RDKitMol',
                         conf_id: int,
                         **kwargs):
            return mol.ToXYZ(conf_id)
        ```

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
            conf_id (int): The index of the conformer to be operated on.

        Returns:
            str: The content of the input file.

        Raises:
            NotImplementedError: If the function is not implemented in the child class.
        """
        raise NotImplementedError

    def write_input_file(self,
                         mol: 'RDKitMol',
                         **kwargs):
        """
        The default input writer of the task. This function iterates through run_ids,
        and call input_writer for each subtask. If failed to write the input file for a
        subtask, the subtask will be marked as failed in mol.keep_ids. The run_ids will be
        updated after the input files are written.

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
                      f' subtask {cid}: {exc}')
                traceback.print_exc()

        # update run_ids so job failed in input generation won't run in later steps
        self.update_run_ids(mol=mol)

    def get_execute_command(self,
                            subtask_id: int,
                            ) -> List[str]:
        """
        Get the command to execute the subtask. The command should be a list of strings,
        where each string is a part of the command. E.g., ['g16', 'input.gjf']. The command
        will be executed in the subtask directory by subprocesses.

        For developers, This function should be implemented in the child class. It should
        return the command to execute the subtask. The subtask_id is the index of the subtask
        to be executed, used for obtaining the path of the input file, etc. This method can be
        left as it is if the task does not need to be executed by subprocesses.

        Args:
            subtask_id (int): The index of the subtask to be executed. Used for obtaining the
                              path of the input file, etc.

        Returns:
            List[str]: The command to execute the subtask.

        Raises:
            NotImplementedError: If the function is not implemented in the child class.
        """
        raise NotImplementedError

    def update_paths(self):
        """
        Update the paths of the subtasks. This function creates the subtask directories
        and the paths of the files in the subtask directories according to the `subtask_dir_name`
        and the `files` defined in the class.

        For developers: This function is designed to be called in the pre-run function of the task.
        """
        # create subtask directories
        if self.singleshot_subtask:
            subtask_dirs = {0: osp.join(self.work_dir, self.subtask_dir_name)}
            os.makedirs(subtask_dirs[0], exist_ok=True)
        else:
            subtask_dirs = {subtask_id: osp.join(self.work_dir, f'{self.subtask_dir_name}{subtask_id}')
                            for subtask_id in self.run_ids}
            for subtask_dir in subtask_dirs.values():
                os.makedirs(subtask_dir, exist_ok=True)
        self.paths = {'subtask_dir': subtask_dirs}
        # create file paths
        for ftype, fname in self.files.items():
            self.paths[ftype] = {subtask_id: osp.join(subtask_dir, fname)
                                 for subtask_id, subtask_dir in subtask_dirs.items()}

    def runner(self,
               subtask_id: int,
               **kwargs):
        """
        The runner of each subtask. The default implementation is to execute the command
        (get from get_execute_command) via subprocess. The working directory of the subprocess
        is the working directory. The stdout is redirected to the log file.

        For developers: This function might be re-implemented in the child class. It can
        also changed to use other methods to run the subtask, e.g., using the python API of
        the program. However, the default implementation should be sufficient for most cases.
        The default implementation return None, and that will be the `subtask_result` argument
        of the `analyze_subtask_result` function. To prepare a parallelized version of the task,
        It is recommended to re-implement the parent `run` function.

        Args:
            subtask_id (int): The index of the subtask to be executed.

        Returns:
            None: This will be passed to the `analyze_subtask_result`.
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
        Basically, the function should read the output file and update the mol object accordingly.
        E.g., read the gaussian log file of subtask N, extract the optimized geometry and energy, and
        update the mol object with the optimized geometry and energy.

        For developers: This function should be implemented in the child class. The subtask_result
        is the return value of the `runner` function. In this function, attributes mentioned in the
        `init_attrs` are suggested to be updated; `keep_ids` should be updated as well to mark the
        subtask's status. All errors raised in this function will be caught and printed in the upper level
        function `run`.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
            subtask_id (int): The index of the subtask to be executed.
            subtask_result (Any): The result of the subtask. This is the return value of the `runner` function.
            **kwargs: Other keyword arguments.

        Raises:
            NotImplementedError: If the function is not implemented in the child class.
        """
        raise NotImplementedError

    def pre_run(self,
                mol: 'RDKitMol',
                **kwargs):
        """
        The preprocessing steps before running the actual task (e.g., gaussian calculation, defined in `run`).
        The default implementation involves the following steps:
            1. Set the number of conformers to be optimized to n_subtasks.
            2. Create the directories and paths for the subtasks.
            3. Write the input files for the subtasks.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
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
    def run(self,
            mol: 'RDKitMol',
            **kwargs):
        """
        Run the task on the molecule. This conducts the most important work of the task.
        The default implementation involves the following steps:
            1. Update the molecule information (e.g., create a copy of the molecule if necessary)
            2. Update the multiplicity and charge of the molecule
            3. Run the subtasks in sequence and analyze the results of each subtask
        The subtasks are defined in the `run_ids` attribute of the task and executed in sequence by
        the `runner` function. The results of the subtasks are analyzed in the `analyze_subtask_result`.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
            kwargs: Other keyword arguments.

        Returns:
            'RDKitMol': The molecule (or its copy) with updated attributes.

        For developers: If you want to re-implement this function, it is recommended to keep the first two
        steps and only change the third step.
        """
        # Set up molecule information
        # 1. Update the molecule information
        # 2. Update the multiplicity and charge of the molecule
        new_mol = self.update_mol(mol=mol)
        kwargs['mult'], kwargs['charge'] = \
                self._get_mult_and_chrg(mol=mol,
                                        multiplicity=kwargs.get('multiplicity'),
                                        charge=kwargs.get('charge'))

        # 4. Run the subtasks in sequence
        run_ids = self.run_ids if not self.singleshot_subtask else [0]
        for subtask_id in run_ids:

            # 4.1. run the subtask defined in runner
            try:
                subtask_result = self.runner(mol=mol,
                                             subtask_id=subtask_id,
                                             **kwargs)
            except Exception as exc:
                new_mol.keep_ids[subtask_id] = False
                print(f'Error in running subtask {subtask_id} of {self.label}: {exc}')
                traceback.print_exc()
                continue

            # 4.2. analyze the subtask result
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
        The postprocessing steps after running the actual task (e.g., gaussian calculation, defined in `run`).
        The default implementation involves the following steps:
            1. Update the number of successful subtasks
            2. Clean up working directory
        """
        # 1. Update the number of successful subtasks
        self.update_n_success()

        # 2. Clean up working directory
        self.clean_work_dir()
