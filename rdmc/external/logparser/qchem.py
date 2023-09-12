#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import re

from rdmc import RDKitMol
from rdmc.external.logparser.base import CclibLog


class QChemLog(CclibLog):
    """
    A class helps to parse the Gaussian log files and provides information
    helpful to analyze the calculation.
    """

    opt_critieria = ['Gradient', 'Displacement', 'Energy change']
    time_regex = r'[a-zA-Z]+\s+\d+\s+\d{2}\:\d{2}\:\d{2}\s+\d{4}'
    _label = 'qchem'

    def _update_status(self):
        """
        Update the status of the calculation. Reference: arc/job/trsh.py

        # Todo: Add cases for other types of termination.
        """
        self._success = False
        self._finished = False
        self._termination_time = None

        with open(self.path) as f:
            reverse_lines = f.readlines()[::-1]

        for i, line in enumerate(reverse_lines):
            if 'Thank you very much for using Q-Chem.  Have a nice day.' in line:
                self._finished = True
                time_str = re.search(self.time_regex, reverse_lines[i + 4]).group()
                for l in reverse_lines[i + 1:]:
                    if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in l:
                        self._success = False
                        break
                else:
                    self._success = True
                    self._termination_time = datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                break
            elif 'SCF failed' in line:
                self._finished = True
                break
            elif 'Invalid charge/multiplicity' in line:
                self._finished = True
                break

    def _update_job_type(self):
        """
        Update the job type.
        """
        scheme_lines = []
        with open(self.path, 'r') as f:
            read = False
            for line in f:
                line = line.lower()
                if '$rem' in line:
                    read = True
                elif '$end' in line:
                    read = False
                if read and 'jobtype' in line:
                    scheme_lines.append(line.split()[-1])
        schemes = ''.join(scheme_lines)

        job_type = []
        if 'opt' in schemes:
            job_type.append('opt')
        if 'ts' in schemes:
            job_type.append('ts')
        if 'freq' in schemes:
            job_type.append('freq')
        if 'rpath' in schemes:
            job_type.append('irc')
        if 'pes_scan' in schemes:
            job_type.append('scan')
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
        elif 'ts' in self.job_type:
            self._is_ts = True
        elif 'freq' in self.job_type and self.num_neg_freqs == 1:
            self._is_ts = True

    # Todo: Check a few implementation
    # Below implemented Dr. Shih-Cheng Li in original QChemLog
    # As simplified version compared to CclibLog
    # Xiaorui hasn't tested whether the full version in CclibLog works

    def get_scf_energies(self,
                         relative: bool = False):
        """
        Get SCF energies in kcal/mol.

        Args:
            relative (bool): Only return the value relative to the minimum. Defaults to ``False``.
        Returns:
            np.array: The SCF energies.
        """
        scf_energies = self.cclib_results.scfenergies

        # Convert from eV to hartree to kcal/mol
        scf_energies = scf_energies / 27.211386245988 * 627.5094740631

        if relative:
            scf_energies -= scf_energies.min()

        return scf_energies

    def get_mol(self,
                refid: int = -1,
                embed_conformers: bool = True,
                neglect_spin: bool = True,
                backend: str = 'openbabel',
                sanitize: bool = True,
                ) -> 'RDKitMol':
        """
        Perceive the xyzs in the file and turn the geometries to conformers.

        Args:
            refid (int): The conformer ID in the log file to be used as the reference for mol perception.
                         Defaults to -1, meaning it is determined by the following criteria:
                         - For opt, it is the last geometry if succeeded; otherwise, the initial geometry;
                         - For freq, it is the geometry input;
                         - For scan, it is the geometry input;
                         - For IRC, uses the initial geometry.
            embed_confs (bool): Whether to embed intermediate conformers in the file to the mol.
                                Defaults to ``True``. To clear, at least one conformer will be included in
                                obtained mol, and its geometry is determined by `refid`.
            neglect_spin (bool): Whether to neglect the error when spin multiplicity are different
                                 between the generated mol and the value in the output file. This
                                 can be useful for calculations involves TS. Defaults to ``True``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
            sanitize (bool): Whether to sanitize the generated mol. Defaults to `True`.
                             If a TS involved in the job, better to set it `False`

        Returns:
            RDKitMol: a molecule generated from the output file.
        """
        if refid in [-1, 0]:
            # Currently -1 and 0 point to the same geometry in the backend cclib,
            # the case of refid = 0 is a bit confusing, since it doesn't point to
            # the first geometry of the job. we use `- num_all_geoms` to query the initial geometry

            if not self.success:
                # Use the starting geometry as the reference geometry if the job doesn't succeed
                refid = -self.num_all_geoms

            # successful jobs
            # uses the last geometries (with refid=-1 and 0)
            # for freq and irc, it is the same geometry as the input geometry
            # for other types, it is the last converged geometry

            elif 'scan' in self.job_type:
                # Use the starting geometry as the reference geometry for scan Jobs.
                refid = -self.num_all_geoms

        # Get the xyz of the conformer at `refid`
        try:
            xyz = self.cclib_results.writexyz(indices=refid)
        except IndexError:
            raise ValueError(f'The provided refid {refid} is invalid.')

        # Convert xyz to mol
        mol = RDKitMol.FromXYZ(xyz, backend=backend, sanitize=sanitize)

        # Correct multiplicity if possible
        if mol.GetSpinMultiplicity() != self.multiplicity:
            mol.SaturateMol(multiplicity=self.multiplicity)
        if mol.GetSpinMultiplicity() != self.multiplicity and not neglect_spin:
            raise RuntimeError('Cannot generate a molecule with the exact multiplicity in the output file')

        # Embedding intermediate conformers
        if embed_conformers:
            num_confs = self.num_all_geoms
            mol.EmbedMultipleNullConfs(n=num_confs)
            for i in range(num_confs):
                mol.SetPositions(coords=self.cclib_results.atomcoords[i], id=i)
        return mol
