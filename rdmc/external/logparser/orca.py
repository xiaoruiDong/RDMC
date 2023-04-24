#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

from rdmc.external.logparser.base import CclibLog


class ORCALog(CclibLog):
    """
    A class helps to parse the Gaussian log files and provides information
    helpful to analyze the calculation.

    # Todo:
    Test this class. most implementation hasn't been tested yet.
    """

    time_regex = r''  # Orca 4 doesn't have time stamp in the log file; need to check orca 5
    opt_criteria = ['TolE', 'TolRMSG', 'TolMaxG', 'TolRMSD', 'TolMaxD']
    _label = 'orca'

    def _update_status(self):
        """
        Update the status of the calculation. Reference: arc/job/trsh.py
        """
        self._success = False
        self._finished = False
        self._termination_time = None

        with open(self.path) as f:
            lines = f.readlines()

        # A very simplified version of orca troubleshooting in ARC
        for line in lines[::-1]:
            if 'ORCA TERMINATED NORMALLY' in line:
                self._success = True
                self._finished = True
                break
            elif 'ORCA finished by error termination' in line:
                self._finished = True
                break
            elif 'Error : multiplicity' in line:
                self._finished = True
                break
            elif 'UNRECOGNIZED OR DUPLICATED KEYWORD' in line:
                self._finished = True
                break
            elif 'There are no CABS' in line:
                self._finished = True
                break
            elif 'This wavefunction IS NOT FULLY CONVERGED!' in line:
                self._finished = True
                break

    def _update_schemes(self):
        """
        Update the job type.
        """
        scheme_lines = []
        with open(self.path, 'r') as f:
            read = False
            for line in f:
                if 'INPUT FILE' in line:
                    read = True
                elif read:
                    try:
                        inp_line = '> '.join(line.split('> ')[1:])
                    except IndexError:  # Invalid line
                        continue
                    if inp_line.startswith('*'):  # geometry
                        read = False
                        break
                    scheme_lines.append(inp_line.lower())

        simple_schemes = {'energy': 'sp',
                          'sp': 'sp',
                          'opt': 'opt',
                          'optts': 'opt',
                          'irc': 'irc',
                          'freq': 'freq',
                          'numfreq': 'freq'}

        schemes = {}
        for line in scheme_lines:
            read_block = False
            current_block = ''
            if line.startswith('!'):
                # Simple notation
                for item in line.split():
                    scheme_name = simple_schemes.get(item)
                    if scheme_name and not schemes.get(scheme_name):
                        schemes[scheme_name] = {}
                    if item == 'optts':
                        schemes[scheme_name].update({'ts': None})
            elif line.startswith('%maxcore'):
                schemes['memory'] = float(line.split()[1])
            elif line.startswith('%'):
                current_block = line.split()[0][1:]
                read_block = True
                if not schemes.get(current_block):
                    schemes[current_block] = {}
            elif read_block:
                items = line.strip().split('#')[0].split()
                if len(items) == 1:
                    schemes[current_block][items[0]] = None
                elif len(items) == 2:
                    schemes[current_block][items[0]] = items[1]
                else:
                    print('Warning: cannot parse the line: {}'.format(line))
        self._schemes = schemes

    def _update_job_type(self):
        """
        Update the job type.

        # Todo: Need a way to check scan
        """
        schemes = self.schemes

        job_type = []
        if 'opt' in schemes:
            job_type.append('opt')
        if 'freq' in schemes:
            job_type.append('freq')
        if 'irc' in schemes:
            job_type.append('irc')
        if 'sp' in schemes:
            job_type.append('sp')
        if len(job_type) == 0:
            job_type.append('sp')
        self._job_type = job_type

    def _update_is_ts(self):
        """
        Update the information of transition state.
        """
        self._is_ts = False
        if 'irc' in self.job_type:
            self._is_ts = True
        elif 'opt' in self.schemes and 'ts' in self.schemes['opt']:
            self._is_ts = True
        elif 'freq' in self.job_type and self.num_neg_freqs == 1:
            self._is_ts = True

    @property
    def multiplicity(self):
        """
        Get the multiplicity of the molecule.

        # For unknown reason sometimes multiplicity is not parsed correctly by cclib
        """
        try:
            return self.cclib_results.mult
        except AttributeError:
            # 'ccData_optdone_bool' object has no attribute 'multiplicity'
            # Try to read from the input section
            read = False
            with open(self.path, 'r') as f:
                for line in f:
                    if 'INPUT FILE' in line:
                        read = True
                    elif read and '****END OF INPUT****' in line:
                        # End of the input section
                        read = False
                        break
                    elif read:
                        try:
                            # line example: | 16> * xyz 0 1
                            return int(line.split('> *')[1].strip().split()[2])
                        except IndexError:  # Invalid line
                            continue
            raise RuntimeError('Cannot find multiplicity in the log file.')
