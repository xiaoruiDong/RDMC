#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task import QChemBaseTask
from rdmc.conformer_generation.verifier.freq import FreqVerifier


class QChemFreqVerifier(FreqVerifier, QChemBaseTask):
    """
    The class for verifying the species or TS by calculating and checking its frequencies using QChem.
    Since frequency may be calculated in an previous job. The class will first check if frequency
    results are available. If not, it will launch jobs to calculate frequencies.

    Args:
        method (str, optional): The method (level of theory) available in QChem to be used.
                                defaults to wb97x-d3.
        basis (str, optional): The basis set to use. Defaults to def2-svp.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        cutoff_freq (float, optional): A cutoff frequency determine whether a imaginary frequency
                                       is a valid mode. Only used for TS verification. Defaults to -100 cm^-1,
                                       that is imaginary frequencies between -100 to 0 cm^-1 are
                                       considered not valid reaction mode.
    """

    subtask_dir_name = 'qchem_freq'
    files = {'input_file': 'qchem_freq.gjf',
             'log_file': 'qchem_freq.log'}
    keep_files = ['qchem_freq.gjf', 'qchem_freq.log']

    def task_prep(self,
                  **kwargs,
                  ):
        """
        Set the method, number of processors, memory and cutoff.

        Args:
            method (str, optional): The method (level of theory) available in QChem to be used.
                                    defaults to wb97x-d3.
            basis (str, optional): The basis set to use. Defaults to def2-svp.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            cutoff_freq (float, optional): A cutoff frequency determine whether a imaginary frequency
                                        is a valid mode. Only used for TS verification. Defaults to -100 cm^-1,
                                        that is imaginary frequencies between -100 to 0 cm^-1 are
                                        considered not valid reaction mode.
        """
        super().task_prep(**kwargs)
        super(FreqVerifier, self).task_prep(**kwargs)

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the frequencies
        from the QChem log file and set them to the molecule.
        """
        if self.need_calc_freqs(mol, subtask_id):
            log = self.logparser(self.paths['log_file'][subtask_id])
            # 1. Parse frequencies
            if log.success:
                mol.frequencies[subtask_id] = log.freqs
            else:
                mol.keep_ids[subtask_id] = False
                print(f'Unsuccessful freq calculation of the geometry of conformer {subtask_id} in {self.label}')
                return
        mol.keep_ids[subtask_id] = self.check_negative_freqs(freqs=mol.frequencies[subtask_id],
                                                             ts=kwargs.get('ts') or False)
