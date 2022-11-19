#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to read QChem output file.
"""

import datetime
import re
from typing import Optional

import numpy as np
import cclib.io

from rdmc import RDKitMol


TERMINATION_TIME_REGEX = r'[a-zA-Z]+\s+\d+\s+\d{2}\:\d{2}\:\d{2}\s+\d{4}'

OPT_CRITERIA = ['Gradient', 'Displacement', 'Energy change']


class QChemLog(object):
    """
    A class helps to parse the QChem log files and provides information
    helpful to analyze the calculation.
    """
    time_regex = TERMINATION_TIME_REGEX
    opt_criteria = OPT_CRITERIA

    def __init__(self,
                 path: str,
                 job_type: Optional[list] = None,
                 ts: Optional[bool] = None):
        """
        Initiate the qchem log parser instance.
        Args:
            path (str): The path to the qchem log file.
            job_type (list, optional): In case the scheme parser doesn't work properly, you could manually set the job type.
                                       Available options are: `sp`, `opt`, `freq`, `irc`, `scan`. Assigning multiple job types
                                       is allowed. Defaults to None, no manual assignment.
            ts (bool, optional): In case the scheme parser doesn't work properly, you could manually set whether the job involves a TS.
                                       Available options are `True` and `False`. Defaults to None, no manual assignment.
        """
        self.path = path
        if job_type:
            self._job_type = job_type
        if ts is not None:
            self._ts = ts

    @property
    def success(self):
        """
        Check if the job succeeded.
        """
        return self._get_val_and_auto_update(
            var_name='_success',
            update_fun=self._update_status,
        )

    def _get_val_and_auto_update(self,
                                 var_name: str,
                                 update_fun,):
        """
        This is a method help to update an attribute automatically
        by calling its corresponding update function.
        Args:
            var_name (str): The variable name to get or update
            update_fun (str): The method name for the update function.
        """
        try:
            var = getattr(self, var_name)
        except AttributeError:
            update_fun()
            var = getattr(self, var_name)
        return var

    ###################################################################
    ####                                                           ####
    ####                        Basic attributes                   ####
    ####                                                           ####
    ###################################################################

    @property
    def success(self):
        """
        Check if the job succeeded.
        """
        return self._get_val_and_auto_update(
            var_name='_success',
            update_fun=self._update_status,
        )

    def get_termination_time(self):
        """
        Get the termination time in datetime.datetime format.
        """
        if not self.success:
            return None
        return self._get_val_and_auto_update(
            var_name='_termination_time',
            update_fun=self._update_status,
        )


    def _update_status(self):
        with open(self.path) as f:
            lines = f.readlines()[-15:]
        for i, line in enumerate(lines):
            if 'Thank you very much for using Q-Chem.  Have a nice day.' in line:
                self._success = True
                time_str = re.search(self.time_regex, lines[i-4]).group()
                self._termination_time = datetime.datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                return
        self._success = False
        self._termination_time = None

    @property
    def job_type(self):
        """
        Get the job type.
        """
        return self._get_val_and_auto_update(
            var_name='_job_type',
            update_fun=self._update_job_type,
        )

    def _update_job_type(self):
        """
        Update the job type.
        """
        scheme_lines = []
        with open(self.path) as f:
            read =False
            for line in f:
                line = line.lower()
                if '$rem' in line:
                    read = True
                elif '$end' in line:
                    read =False
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

    @property
    def is_ts(self):
        """
        If the job involves a TS calculation.
        """
        return self._get_val_and_auto_update(
            var_name='_ts',
            update_fun=self._update_ts,
        )

    def _update_ts(self):
        """
        Update the ts property.
        """
        if 'irc' in self.job_type:
            self._ts = True
            return

        if 'ts' in self.job_type:
            self._ts = True
            return

        if 'freq' in self.job_type and self.num_neg_freqs == 1:
            self._ts = True
            return

        self._ts = False


    ###################################################################
    ####                                                           ####
    ####                   Data from CClib                         ####
    ####                                                           ####
    ###################################################################

    @property
    def cclib_results(self):
        return self._get_val_and_auto_update(
            var_name='_cclib_results',
            update_fun=self._load_cclib_results,
        )

    def _load_cclib_results(self):
        """
        Parse the results using the cclib package.
        """
        self._cclib_results = cclib.io.ccread(self.path)

    @property
    def multiplicity(self):
        """
        The multiplicity of the molecule

        Return:
            int
        """
        return self.cclib_results.mult

    @property
    def charge(self):
        """
        The charge of the molecule.

        Return:
            int
        """
        return self.cclib_results.charge

    @property
    def freqs(self):
        """
        Return the frequency as a numpy array.
        """
        return getattr(self.cclib_results, 'vibfreqs', None)

    @property
    def neg_freqs(self):
        """
        Return the imaginary frequency as a numpy array.
        """
        try:
            return self.freqs[self.freqs < 0]
        except TypeError:
            raise RuntimeError('This is not a frequency job.')

    @property
    def num_neg_freqs(self):
        """
        Return the number of imaginary frequencies.

        Returns:
            int
        """
        try:
            return np.sum(self.freqs < 0)
        except TypeError:
            raise RuntimeError('This is not a frequency job.')

    @property
    def all_geometries(self):
        """
        Return all geometries in the file as a numpy array.

        Returns:
            np.ndarray
        """
        return self.cclib_results.atomcoords

    @property
    def num_all_geoms(self):
        """
        Return the number of all geometries.

        Returns:
            int
        """
        return self.cclib_results.atomcoords.shape[0]

    @property
    def converged_geometries(self):
        """
        Return converged geometries as a numpy array.

        Returns:
            np.ndarray
        """
        return self.cclib_results.converged_geometries[-1:]

    @property
    def num_converged_geoms(self):
        """
        Return the number of converged geometries. Useful in IRC and SCAN jobs.

        Returns:
            int
        """
        return self.converged_geometries.shape[0]

    @property
    def initial_geometry(self):
        """
        Return the initial geometry of the job. For scan jobs and ircs jobs.
        Intermediate guesses will not be returned.
        """
        return self.cclib_results.new_geometries[0]

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

    ###################################################################
    ####                                                           ####
    ####                      Molecule Operation                   ####
    ####                                                           ####
    ###################################################################

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

    ###################################################################
    ####                                                           ####
    ####                         Result Analysis                   ####
    ####                                                           ####
    ###################################################################

    def get_irc_midpoint(self):
        """
        Find the geometry index of an IRC job where the direction of the path changes.
        """
        scf_e = self.get_scf_energies()
        e_diff = np.diff(scf_e)
        if np.alltrue(e_diff < 0):
            # There is no midpoint
            return
        return np.argmax(e_diff) + 1


# TODO: Let the input format be more flexible
def write_qchem_ts_opt(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1):

    qchem_ts_opt_input = (f'$rem\n'
                          f'JOBTYPE FREQ\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
                          f'$molecule\n'
                          f'{mol.GetFormalCharge()} {mult}\n'
                          f'{mol.ToXYZ(header=False, confId=confId)}\n'
                          f'$end\n\n'
                          f'@@@\n\n'
                          f'$molecule\n'
                          f'read\n'
                          f'$end\n\n'
                          f'$rem\n'
                          f'JOBTYPE TS\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_GUESS READ\n'
                          f'GEOM_OPT_HESSIAN READ\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'GEOM_OPT_MAX_CYCLES 100\n'
                          f'GEOM_OPT_TOL_GRADIENT 100\n'
                          f'GEOM_OPT_TOL_DISPLACEMENT 400\n'
                          f'GEOM_OPT_TOL_ENERGY 33\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
                          f'@@@\n\n'
                          f'$molecule\n'
                          f'read\n'
                          f'$end\n\n'
                          f'$rem\n'
                          f'JOBTYPE FREQ\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
    )
    return qchem_ts_opt_input


def write_qchem_opt(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1):

    qchem_opt_input = (f'$rem\n'
                      f'JOBTYPE OPT\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'GEOM_OPT_MAX_CYCLES 100\n'
                      f'GEOM_OPT_TOL_GRADIENT 100\n'
                      f'GEOM_OPT_TOL_DISPLACEMENT 400\n'
                      f'GEOM_OPT_TOL_ENERGY 33\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
                      f'$molecule\n'
                      f'{mol.GetFormalCharge()} {mult}\n'
                      f'{mol.ToXYZ(header=False, confId=confId)}\n'
                      f'$end\n\n'
                      f'@@@\n\n'
                      f'$molecule\n'
                      f'read\n'
                      f'$end\n\n'
                      f'$rem\n'
                      f'JOBTYPE FREQ\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_GUESS READ\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
    )
    return qchem_opt_input


def write_qchem_irc(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1):

    qchem_irc_input = (f'$rem\n'
                      f'JOBTYPE FREQ\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
                      f'$molecule\n'
                      f'{mol.GetFormalCharge()} {mult}\n'
                      f'{mol.ToXYZ(header=False, confId=confId)}\n'
                      f'$end\n\n'
                      f'@@@\n\n'
                      f'$molecule\n'
                      f'read\n'
                      f'$end\n\n'
                      f'$rem\n'
                      f'JOBTYPE RPATH\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_GUESS READ\n'
                      f'GEOM_OPT_HESSIAN READ\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
    )
    return qchem_irc_input
