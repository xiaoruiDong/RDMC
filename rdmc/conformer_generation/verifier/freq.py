#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from rdmc.conformer_generation.task import MolIOTask


class FreqVerifier(MolIOTask):
    """
    The base frequency verifier. Useful if:
    - freqs are calculated in a previous task (e.g., optimization)
    - creating a more specific new task involving frequency calculation

    Args:
        cutoff_freq (float, optional): A cutoff frequency determine whether a imaginary frequency
                                       is a valid mode. Only used for TS verification. Defaults to -100 cm^-1,
                                       that is imaginary frequencies between -100 to 0 cm^-1 are
                                       considered not valid reaction mode.
    """

    create_mol_flag = False
    calc_type = "freq"

    def task_prep(self,
                  cutoff_freq: float = -100.,
                  **kwargs,
                  ):
        """
        Set the frequency cutoff.

        Args:
        cutoff_freq (float, optional): A cutoff frequency determine whether a imaginary frequency
                                       is a valid mode. Only used for TS verification. Defaults to -100 cm^-1,
                                       that is imaginary frequencies between -100 to 0 cm^-1 are
                                       considered not valid reaction mode.
        """
        self.cutoff_freq = cutoff_freq

    def need_calc_freqs(self,
                        mol: 'RDKitMol',
                        subtask_id: int,
                        ) -> bool:
        """
        Check if the frequency verification should proceed, including:
            1. Skip Uni-atom molecules that don't have frequencies.
            2. molecules whose frequencies has been calculated in previous jobs.

        For developers: This method is designed to be called before the actual runner.

        Args:
            mol ('RDKitMol') The molecule.

        Returns:
            bool: Whether the frequency verification should proceed.
        """
        # Uni-atom molecules don't have frequencies
        if mol.GetNumAtoms() == 1:
            return False
        # Frequencies has been calculated in previous jobs
        if mol.frequencies[subtask_id] is not None:
            return False
        return True

    def check_negative_freqs(self,
                             freqs: np.ndarray,
                             ts: bool,
                             ) -> bool:
        """
        Check if the number of negative frequencies is correct.
        0 for non-TSs, and 1 for TSs. For TSs, the valid negative frequency
        is determined by the cutoff frequency (self.cutoff_freq).

        Args:
            freqs (np.ndarray): The frequencies in one dimensional np.array.
        Returns:
            bool: Whether the number of negative frequencies are reasonable.
        """
        if ts:
            return sum(freqs < self.cutoff_freq) == 1
        else:
            return not np.any(freqs < 0)

    def pre_run(self, mol, **kwargs):
        """
        The preprocessing steps before running the actual task (e.g., gaussian calculation, defined in `run`).
        The default implementation involves the following steps:
            1. Set the number of conformers to be optimized to n_subtasks. (from parent)
            2. Create the directories and paths for the subtasks. (from parent)
            3. Write the input files for the subtasks. (from parent)
            4. Initiate frequencies if not available. (new)

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
        """
        super().pre_run(mol, **kwargs)

        # Note, init_attrs is not utilized in freq verifiers to avoid
        # overwriting the frequencies calculated in previous jobs.
        if not hasattr(mol, 'frequencies'):
            mol.frequencies = [None] * mol.GetNumConformers()

    def runner(self,
               mol: 'RDKitMol',
               subtask_id: int,
               **kwargs):
        """
        The runner of each subtask. The default implementation is first to check if there is a need to run
        frequency calculations, and, if so, to execute the command (get from get_execute_command) 
        via subprocess. The working directory of the subprocess is the working directory.
        The stdout is redirected to the log file.

        For developers: This function might be re-implemented in the child class. It can
        also changed to use other methods to run the subtask, e.g., using the python API of
        the program. However, the default implementation should be sufficient for most cases.
        The default implementation return None, and that will be the `subtask_result` argument
        of the `analyze_subtask_result` function. To prepare a parallelized version of the task,
        It is recommended to re-implement the parent `run` function.

        Args:
            mol (RDKitMol): The molecule with conformers to be operated on.
            subtask_id (int): The index of the subtask to be executed.

        Returns:
            None: This will be passed to the `analyze_subtask_result`.
        """
        if self.need_calc_freqs(mol=mol, subtask_id=subtask_id):
            # MolIOTask
            # which is the subprocess executor
            return super().runner(mol=mol, subtask_id=subtask_id, **kwargs)
