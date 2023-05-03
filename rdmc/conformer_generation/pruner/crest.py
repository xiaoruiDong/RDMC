#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import subprocess
import traceback
from typing import List

from rdmc.conformer_generation.task import MolIOTask
from rdmc.conformer_generation.utils import _software_available
from rdmc.external.xtb_tools.utils import CREST_BINARY, XTB_ENV

_software_available['crest'] = osp.isfile(CREST_BINARY)


class CRESTPruner(MolIOTask):
    """
    The class to run CREST calculations to prune conformers. It utilizes the CREGEN
    soritng algorithm to prune conformers.

    Args:
    ewin (float, optional): Set the energy threshold in kcal/mol. Defaults to 0.15 kcal/mol.
    rthr (float, optional): Set RMSD threshold in Ångström. Defaults to 0.125 Å.
    ethr (float, optional): Set energy threshold between conformer pairs in kcal/mol. Defaults to 0.05 kcal/mol.
    bthr (float, optional): Set lower bound for the rotational constant threshold. Defaults to 0.01 (= 1%).
                            The threshold is dynamically adjusted between this value and 2.5%, based on an anisotropy
                            of the rotational constants in the enesemble.
    """

    request_external_software = ['crest']
    keep_files = ['xtb.log', 'confs.xyz', 'conf_0.xyz', 'enso.tags']
    files = {'log_file': 'xtb.log',
             'confs_file': 'confs.xyz',
             'conf_0_file': 'conf_0.xyz',
             'enso_tag': 'enso.tags'}
    subtask_dir_name = 'crest'
    singleshot_subtask = True

    def task_prep(self,
                  ewin: float = 1000.0,
                  rthr: float = 0.125,
                  ethr: float = 0.15,
                  bthr: float = 0.01,
                  **kwargs,
                  ):
        """
        Set up the calculation arguments for CREST.

        Args:
            ewin (float, optional): Set the energy threshold in kcal/mol. Defaults to 0.15 kcal/mol.
            rthr (float, optional): Set RMSD threshold in Ångström. Defaults to 0.125 Å.
            ethr (float, optional): Set energy threshold between conformer pairs in kcal/mol. Defaults to 0.05 kcal/mol.
            bthr (float, optional): Set lower bound for the rotational constant threshold. Defaults to 0.01 (= 1%).
                                    The threshold is dynamically adjusted between this value and 2.5%, based on an anisotropy
                                    of the rotational constants in the enesemble.
        """
        self.ethr = ethr
        self.rthr = rthr
        self.bthr = bthr
        self.ewin = ewin
        super().task_prep(**kwargs)

    def write_input_file(self,
                         mol: 'RDKitMol',
                         **kwargs):
        """
        Write the input file for CREST.
        For developers: In Crest two files are required: a .xyz file containing the conformers (type: confs_file)
        and a geometry file (used for topology check, type: conf_0_file). In this implementation, the .xyz file for
        the first conformer is written to conf_0_file, and the .xyz file for all conformers is written to confs_file.

        Args:
            mol (RDKitMol): The molecule to be pruned.
        """
        with open(self.paths['conf_0_file'][0], 'w') as f:
            f.write(mol.ToXYZ(header=True))

        if not hasattr(mol, 'energies'):
            # Only pruning based on coordinates
            energies = [0.0] * len(mol.GetConformers())
            print('Warning: No energies are provided for the molecule. Only '
                  'pruning based on coordinates.')
        else:
            energies = mol.energies

        with open(self.paths['confs_file'][0], 'w') as f:
            f.write('\n'.join([mol.ToXYZ(confId=cid,
                                         comment=f"{energies[cid]:.8f} !CONF{cid}")
                               for cid in self.run_ids]))

    def get_execute_command(self,
                            **kwargs,
                            ) -> List[str]:
        """
        Get the command to execute CREST.
        Reference: https://crest-lab.github.io/crest-docs/page/documentation/keywords.html
        """
        return [CREST_BINARY,
                self.paths['conf_0_file'][0],
                '--cregen',
                self.paths['confs_file'][0],
                '--ewin', str(self.ewin),
                '--rthr', str(self.rthr),
                '--ethr', str(self.ethr),
                '--bthr', str(self.bthr),
                '--enso']

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               **kwargs):
        """
        Analyze the result of CREST CREGEN pruning, involving the following steps:
            1. Read and Parse the enso.tags file to get the unique ids.
            2. Update the keep_ids.

        Args:
            mol (RDKitMol): The molecule to be pruned.
        """
        # 1.1 Read the enso.tags file
        with open(self.paths['enso_tag'][0], 'r') as f:
            lines = f.readlines()

        # 1.2 Parse the enso.tags file to get the unique ids.
        unique_ids = []
        for line in lines:
            try:
                # Example;
                #  -22.60920899 !CONF50
                # Every line follows this format
                cid = int(line.strip().split()[-1].split('!CONF')[-1])
            except IndexError:
                continue
            except ValueError as exc:
                print(f'Error in parsing the result of {self.label}: {exc}')
                traceback.print_exc()
                # Do not prune any conformer if there is an error in parsing the result.
                return
            else:
                unique_ids.append(cid)

        # 2. Update the keep_ids.
        for cid in self.run_ids:
            mol.keep_ids[cid] = cid in unique_ids
