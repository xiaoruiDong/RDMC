#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import re

from rdmc.external.logparser.base import CclibLog
from rdmc.external.logparser.utils import scheme_to_dict, COMPOSITE_METHODS


class GaussianLog(CclibLog):
    """
    A class helps to parse the Gaussian log files and provides information
    helpful to analyze the calculation.
    """

    time_regex = r'[a-zA-Z]+\s+\d+\s+\d{2}\:\d{2}\:\d{2}\s+\d{4}'
    opt_criteria = ['Force Maximum', 'Force RMS', 'Displacement Maximum', 'Displacement RMS']
    _label = 'gaussian'

    def _update_status(self):
        """
        Update the status of the calculation.
        """
        # Only check the last 10 lines
        # Usually this is enough to determine the status
        with open(self.path) as f:
            lines = f.readlines()[-10:]

        self._success = False
        self._finished = False
        self._termination_time = None
        for line in lines:
            if 'Normal termination' in line:
                # Example:
                # Normal termination of Gaussian 16 at Thu Oct 28 13:55:59 2021.
                self._success = True
                self._finished = True
                time_str = re.search(self.time_regex, line).group()
                self._termination_time = datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                return
            elif 'Error termination' in line:
                # Example:
                # 1. Error termination request processed by link 9999. (without timestamp)
                # 2. Error termination via Lnk1e in /opt/g16/l9999.exe at Fri Oct 29 14:32:23 2021. (with timestamp)
                self._finished = True
                try:
                    time_str = re.search(self.time_regex, line).group()
                    self._termination_time = datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                    return
                except AttributeError:
                    pass
            elif 'Erroneous write' in line:
                # Example:
                # Erroneous write. Write -1 instead of 198024.
                # fd = 4
                # orig len = 198024 left = 198024
                # g_write
                break

    def _update_job_type(self):
        """
        Update the job type.
        """
        schemes = self.schemes

        if schemes['LOT']['LOT'] in COMPOSITE_METHODS:
            # In Gaussian 16 you can set g4(sp) or g4(noopt)
            # to change the job type, they are processed in the following block
            if 'freq' in schemes and 'sp' in schemes:
                self._job_type = ['freq', 'sp']
            elif 'sp' in schemes:
                self._job_type = ['sp']
            else:
                self._job_type = ['opt', 'freq', 'sp']
            return

        job_type = []
        if 'opt' in schemes:
            for key in schemes['opt'].keys():
                if 'addred' in key or 'modred' in key:
                    # Here, dict.get is not used.
                    # Using loops and `in` is because users may write the keyword differently, e.g., addredundant
                    try:
                        # If this is not a scan, the following function will raises a Runtime Error
                        self.get_scannames()
                    except RuntimeError:
                        continue
                    else:
                        job_type.append('scan')
                    break
            else:
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

    def _update_schemes(self):
        """
        Parse the scheme used by the Gaussian job.
        """
        count = 0
        scheme_lines = []
        with open(self.path) as f:
            for _ in range(500):
                line = f.readline()
                if '--------------------------' in line:
                    count += 1
                if count == 3:
                    scheme_lines.append(line)
                elif count == 4 or line == '':
                    break
        # There is a trivial point: the line may split in the middle of a
        # a word or at a space gap, therefore simply `strip` doesn't work
        scheme_str = ''.join(line.strip('\n')[1:] for line in scheme_lines[1:])

        try:
            self._schemes = scheme_to_dict(scheme_str)
        except Exception as e:
            print(f'Calculation scheme parser encounters a problem. \nGot: {e}\n'
                  f'Feel free to raise an issue about this error at RDMC\'s Github Repo.')
            self._schemes = {}

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
