#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to read Gaussian output file.
"""

import datetime
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cclib.io

from rdmc import RDKitMol
from rdmc.view import mol_viewer, freq_viewer, mol_animation

try:
    from ipywidgets import interact, IntSlider, Dropdown, FloatLogSlider
except ImportError:
    INTERACT = False
else:
    INTERACT = True

# TODO: All information below need to be expanded
SEMIEMPIRICAL = ['pm6', 'am1', 'pm3']
DFT = ['b3lyp', 'cam-b3lyp', 'wb97xd', 'wb97x', 'apfd', 'b97d3', 'blyp', 'm06', 'm062x']
COMPOSITE_METHODS = ['cbs-qb3', 'rocbs-qb3', 'cbs-4m', 'cbs-apno', 'g1', 'g2', 'g3',
                     'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp2',
                     'w1u', 'w1bd', 'w1ro']
HF = ['hf']
MP_METHOD = ['mp2', 'mp3', 'mp4', 'mp4(dq)', 'mp4(sdq)', 'mp5']
DOUBLE_HYBRID = ['b2plyp', 'mpw2plyp', 'b2plypd', 'b2plypd3', 'dsdpbep86',
                 'pbe0dh', 'pbeqidh', 'M08HX', 'MN15', 'MN15L']
COUPLE_CLUSTER = ['ccsd', 'ccsdt', 'ccsd(t)']

METHODS = HF + DFT + MP_METHOD + DOUBLE_HYBRID + COUPLE_CLUSTER
DEFAULT_METHOD = 'hf'

BASIS_SETS = ['sto-3g', '3-21g', '6-21g', '4-31g', '6-31g', '6-311g', 'cbsb7',
              '6-21g*', '6-21g(d)', '6-21g**', '6-21g(d,p)'
              '6-31g*', '6-31g(d)', '6-31g**', '6-31g(d,p)',
              'cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z', 'cc-pv6z',
              'def2sv', 'def2svp', 'def2svpp', 'def2tzv', 'def2tzvp',
              'def2tzvpp', 'def2qzv', 'def2qzvp', 'def2qzvpp']
DEFAULT_BASIS_SET = 'sto-3g'

SCHEME_REGEX = r'(\w+=?\([^\(\)]+\)|\w+=\w+|[^\s=\(\)]+)'
TERMINATION_TIME_REGEX = r'[a-zA-Z]+\s+\d+\s+\d{2}\:\d{2}\:\d{2}\s+\d{4}'

OPT_CRITERIA = ['Force Maximum', 'Force RMS', 'Displacement Maximum', 'Displacement RMS']


class GaussianLog(object):
    """
    A class helps to parse the Gaussian log files and provides information
    helpful to analyze the calculation.
    """

    time_regex = TERMINATION_TIME_REGEX
    opt_criteria = OPT_CRITERIA

    def __init__(self, path: str):
        """
        Initiate the gaussian log parser instance.

        Args:
            path (str): The path to the gaussian log file.
        """
        self.path = path

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

    @property
    def finished(self):
        """
        Check if the job is finished.
        """
        return self._get_val_and_auto_update(
            var_name='_finished',
            update_fun=self._update_status,
        )

    def get_termination_time(self):
        """
        Get the termination time in datetime.datetime format.
        """
        if not self.finished:
            return None
        return self._get_val_and_auto_update(
            var_name='_termination_time',
            update_fun=self._update_status,
        )

    def _update_status(self):
        with open(self.path) as f:
            lines = f.readlines()[-5:]
        for line in lines:
            if 'Normal termination' in line:
                # Example:
                # Normal termination of Gaussian 16 at Thu Oct 28 13:55:59 2021.
                self._success = True
                self._finished = True
                time_str = re.search(self.time_regex, line).group()
                self._termination_time = datetime.datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                return
            elif 'Error termination' in line:
                # Example:
                # 1. Error termination request processed by link 9999. (without timestamp)
                # 2. Error termination via Lnk1e in /opt/g16/l9999.exe at Fri Oct 29 14:32:23 2021. (with timestamp)
                self._success = False
                self._finished = True
                try:
                    time_str = re.search(self.time_regex, line).group()
                except AttributeError:
                    pass
                else:
                    self._termination_time = datetime.datetime.strptime(time_str, '%b %d %H:%M:%S %Y')
                    return
        self._success = False
        self._finished = False
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
        schemes = self.schemes

        if schemes['LOT']['LOT'] in COMPOSITE_METHODS:
            self._job_type = ['opt', 'freq', 'sp']
            return

        job_type = []
        if 'opt' in schemes:
            for key in schemes['opt'].keys():
                if 'addred' in key or 'modred' in key:
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

    @property
    def schemes(self):
        """
        Get the schemes, as a dict, that used in the Gaussian calculation.
        """
        return self._get_val_and_auto_update(
            var_name='_schemes',
            update_fun=self._update_schemes,
        )

    def parse_schemes(self):
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
        scheme = ''.join(line.strip('\n')[1:] for line in scheme_lines[1:])
        return scheme.lower()

    def _update_schemes(self):
        """
        Update the schemes property.
        """
        scheme_str = self.parse_schemes()
        self._schemes = scheme_to_dict(scheme_str)

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

        if 'opt' in self.schemes and 'ts' in self.schemes['opt']:
            self._ts = True
            return

        if 'freqs' in self.job_type and self.num_neg_freqs == 1:
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
        converged_idx = self.get_converged_geom_idx()
        if 'opt' in self.job_type and 'scan' not in self.job_type:
            if np.any(converged_idx):
                return self.cclib_results.atomcoords[converged_idx][-1:]
            else:
                return np.array([])
        if 'irc' in self.job_type or 'scan' in self.job_type:
            return self.cclib_results.atomcoords[converged_idx]
        if self.job_type == ['freq'] and self.success:
            return np.array([self.initial_geometry])
        else:
            return self.cclib_results.converged_geometries[:1]

    @property
    def num_converged_geoms(self):
        """
        Return the number of converged geometries. Useful in IRC and SCAN jobs.

        Returns:
            int
        """
        return self.converged_geometries.shape[0]

    def get_converged_geom_idx(self,
                               as_numbers: bool = False,):
        """
        Return the indexes of geometries that are converged. By default, a
        numpy array of True and False will be returned. But you can output numeric
        results by changing the argument.

        Args:
            as_numbers (bool): Whether returns a list of numbers. Defaults to ``False`` for better performance.

        Returns:
            np.array
        """
        if 'opt' in self.job_type or 'irc' in self.job_type:
            idx_array = self.cclib_results.OPT_DONE & self.cclib_results.optstatus > 0
        elif 'scan' in self.job_type:
            # For IRC and scan, convergence is defined as fulfilling at least 2 criteria for gaussian
            idx_array = self.cclib_results.optstatus >= 2
        else:
            idx_array = np.array([True])
        if 'irc' in self.job_type:
            # Mark the input geometry of the IRC as converged (cclib mark it just as initial geometry '1')
            idx_array[0] = True
        if as_numbers:
            idx_array = np.where(idx_array)[0]
        return idx_array

    @property
    def initial_geometry(self):
        """
        Return the initial geometry of the job. For scan jobs and ircs jobs.
        Intermediate guesses will not be returned.
        """
        return self.cclib_results.new_geometries[0]

    def get_scannames(self,
                   as_list: bool = False,
                   index_0: bool = False):
        """
        Return the scan names (the atom indexes of the internal coordinates).

        Args:
            as_list (bool): Whether return scannames as lists instead of strings.
                            Defaults to ``False``.
            index_0 (bool): In gaussian, atom index starts from 1, while RDKitMol
                            atom index starts from 0. Whether to convert to indexes
                            that starts from 0. Defaults to ``False``.

        Returns:
            list: A string representation of the scanning internal coordinates.
        """
        try:
            orig_scanname = self.cclib_results.scannames
        except AttributeError:
            raise RuntimeError('This is not a scan job!')
        if as_list:
            list_of_scannames = []
            for scanname in orig_scanname:
                if index_0:
                    list_of_scannames.append([int(i) - 1 for i in re.findall(r'\d+', scanname)])
                else:
                    list_of_scannames.append([int(i) for i in re.findall(r'\d+', scanname)])
            return list_of_scannames
        else:
            return orig_scanname

    def get_scanparams(self,
                       converged=True,
                       relative=True,
                       backend='openbabel'):
        """
        Get the values of the scanning parameters. It is assumed that the job is 1D scan.

        Args:
            converged (bool): If only return values for converged geometries.
            relative (bool): If return values that are relative to the first entry.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.

        Returns:
            np.array
        """
        if 'scan' not in self.job_type:
            raise RuntimeError('This is function is only valid for scan job.')
        # Assuming it is a 1D scan
        scan_name = self.get_scannames(as_list=True, index_0=True)[0]
        if converged:
            params = np.array(self.cclib_results.scanparm[0])
        else:
            mol = self.get_mol(converged=False, backend=backend)
            get_val_fun = {2: 'GetBondLength', 3: 'GetAngleDeg', 4: 'GetTorsionDeg'}[len(scan_name)]
            params = np.array([getattr(mol.GetConformer(id=i), get_val_fun)(scan_name)
                                    for i in range(mol.GetNumConformers())])
        if relative:
            params -= params[0]
            if len(scan_name) == 3:
                params[params < 0] += 180.
            elif len(scan_name) == 4:
                params[params < 0] += 360.
        return params

    def get_xyzs(self,
                 converged: bool = True,
                 initial_geom: bool = True,):
        """
        Get the xyz strings of geometries stored in the output file.

        Args:
            converged (bool): Only return the xyz strings for converged molecules.
                              Defaults to True.
            initial_geom (bool): By default cclib_results will replace the geometry 1 (index 0)
                                 with the converged geometry. This options allows to keep the
                                 geometry 1 as the input geometry. Defaults to True.
        """
        if converged:
            converged_idx = self.get_converged_geom_idx(as_numbers=True)
            if 'opt' in self.job_type:
                xyz_strs = [self.cclib_results.writexyz(indices=i) for i in converged_idx[-1:]]
            else:
                xyz_strs = [self.cclib_results.writexyz(indices=i) for i in converged_idx]
        else:
            xyz_strs = [self.cclib_results.writexyz(indices=i) for i in range(self.num_all_geoms)]

        if initial_geom:
            if self.cclib_results.optstatus[0] >= 4 or 'irc' in self.job_type:
                xyz_strs[0] = self.cclib_results.writexyz(indices=-self.num_all_geoms)
            elif 'opt' in self.job_type and not converged:
                xyz_strs[0] = self.cclib_results.writexyz(indices=-self.num_all_geoms)
        return xyz_strs

    def get_scf_energies(self,
                         converged: bool = True,
                         only_opt: bool = False,
                         relative: bool = True):
        """
        Get SCF energies in kcal/mol.

        Args:
            converged (bool): Only get the SCF energies for converged geometries.'
                              Defaults to ``True``.
            only_opt (bool): For composite method like CBS-QB3, you can choose only to
                             return SCF energies only for the optimization step. Defaults
                             to ``False``.
            relative (bool): Only return the value relative to the minimum. Defaults to ``True``.
        """
        num_opt_geoms = self.num_all_geoms
        if only_opt and 'opt' in self.job_type:
            scf_energies = self.cclib_results.scfenergies[:num_opt_geoms]
        else:
            scf_energies = self.cclib_results.scfenergies

        if converged:
            if 'opt' in self.job_type:
                sub1 = scf_energies[:num_opt_geoms][self.get_converged_geom_idx()]
            else:
                sub1 = scf_energies[self.get_converged_geom_idx()]
            sub2 = scf_energies[num_opt_geoms:]
            scf_energies = np.concatenate([sub1, sub2])

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
               converged: bool = True,
               neglect_spin: bool = True,
               backend: str = 'openbabel'
               ) -> 'RDKitMol':
        """
        Perceive the xyzs in the file and turn the geometries to conformers.

        Args:
            refid (int): The id of the conformer to be used as a reference for mol generation.
            embed_confs (bool): Whether to embed intermediate conformers in the file to the mol.
                                Defaults to ``True``.
            converged (bool): Whether to only embed converged conformers to the mol. This option
                              is only valid when ``embed_confs`` is ``True``.
            neglect_spin (bool): Whether to neglect the error when spin multiplicity are different
                                 between the generated mol and the value in the output file. This
                                 can be useful for calculations involves TS. Defaults to ``True``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.

        Returns:
            RDKitMol: a molecule generated from the output file.
        """
        if refid not in [-1, 0]:
            try:
                xyz = self.cclib_results.writexyz(indices=refid)
            except IndexError:
                raise ValueError(f'The provided refid {refid} is invalid.')
        elif 'scan' in self.job_type or 'irc' in self.job_type:
            # Use the starting geometry as the reference geometry for IRC job and Scan Job.
            xyz = self.cclib_results.writexyz(indices=-self.num_all_geoms)
        elif self.success:
            # If succeeded, use the default (converged) geometry
            xyz = self.cclib_results.writexyz()
        else:
            # If not converged, use the initial guess geometry
            xyz = self.cclib_results.writexyz(indices=-self.num_all_geoms)

        mol = RDKitMol.FromXYZ(xyz, backend=backend)

        if mol.GetSpinMultiplicity() != self.multiplicity:
            mol.SaturateMol(multiplicity=self.multiplicity)

        if mol.GetSpinMultiplicity() != self.multiplicity and not neglect_spin:
            raise RuntimeError('Cannot generate a molecule with the exact multiplicity in the output file')

        if embed_conformers and converged:
            mol.EmbedMultipleNullConfs(n=self.num_converged_geoms)
            for i in range(self.num_converged_geoms):
                mol.SetPositions(coords=self.converged_geometries[i], id=i)
        elif embed_conformers:
            num_confs = self.num_all_geoms
            mol.EmbedMultipleNullConfs(n=num_confs)
            for i in range(num_confs):
                mol.SetPositions(coords=self.cclib_results.atomcoords[i], id=i)
        return mol

    def _process_scan_mol(self,
                          converged: bool = True,
                          align_scan: bool = True,
                          align_frag_idx: int = 1,
                          backend: str = 'openbabel'):
        """
        A function helps to process molecule conformers from a scan job.

        Args:
            align_scan (bool): If align the molecule to make the animation cleaner.
                               Defaults to ``True``
            align_frag_idx (int): Value should be either 1 or 2. Assign which of the part to be
                                  aligned.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.

        Returns:
            RDKitMol
        """
        mol = self.get_mol(converged=converged, backend=backend)
        if align_scan:
            # Assume it is 1D scan
            scan_name = self.get_scannames(as_list=True, index_0=True)[0]
            if align_frag_idx == 1:
                atom_map = [(i, i) for i in scan_name[:len(scan_name) - 1]]
            else:
                atom_map = [(i, i) for i in scan_name[-len(scan_name) + 1:]]
            for idx in range(1, mol.GetNumConformers()):
                mol.AlignMol(refMol=mol, prbCid=idx,
                            refCid=0, atomMap=atom_map)
        return mol

    def _process_irc_mol(self,
                         converged: bool = True,
                         bothway: bool = True,
                         backend: str = 'openbabel'):
        """
        A function helps to process molecule conformers from a IRC job.

        Args:
            converged (bool): Whether only process converged conformers. Defaults to ``True``.
            bothway (bool): Whether is a two-way IRC job. Defaults to ``True``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.

        Returns:
            RDKitMol
        """
        # Figure out if there is an 'inverse' point in the IRC path
        midpoint = self.get_irc_midpoint(converged=converged)
        if bothway and midpoint:
            mol = self.get_mol(converged=converged, embed_conformers=False, backend=backend)
            num_confs = self.num_converged_geoms if converged else self.num_all_geoms
            mol.EmbedMultipleNullConfs(n=num_confs)
            coords = self.converged_geometries if converged else self.all_geometries
            # Inverse part of the geometry to make the change in geometry 'monotonically'
            coords[:midpoint] = coords[midpoint-1::-1]
            for i in range(num_confs):
                mol.SetPositions(coords=coords[i], id=i)
        else:
            mol = self.get_mol(converged=converged, backend=backend)
        return mol

    ###################################################################
    ####                                                           ####
    ####                         Result Analysis                   ####
    ####                                                           ####
    ###################################################################

    @property
    def convergence(self):
        """
        Return the geometry convergence criteria and values.

        Returns:
            pd.DataFrame
        """
        return self._get_val_and_auto_update(
            var_name='_convergence',
            update_fun=self._update_convergence_table,
        )

    def _update_convergence_table(self):
        """
        A helper function to create the convergence table.
        """
        if 'opt' not in self.job_type:
            raise RuntimeError('This is not an opt job.')
        data = np.concatenate([self.cclib_results.geotargets.reshape(1, -1), self.cclib_results.geovalues])
        if self.cclib_results.optstatus[0] == 1:
            index=['target'] + list(range(1, self.cclib_results.geovalues.shape[0] + 1))
        self._convergence = pd.DataFrame(data=data, index=index, columns=self.opt_criteria)

    def get_best_opt_geom(self,
                          xyz_str=False):
        """
        Get the best geometry obtained in the optimization job. If the job
        converged, then it is the converged geometries, if not, it is the one
        that is the closest to opt convergence criteria.

        Args:
            xyz_str (bool): Whether to return in numpy array (``False``) or
                            xyz_str (``True``). Defaults to ``False``.
        """
        if 'opt' not in self.job_type:
            raise RuntimeError('This function is only valid for "opt" job type.')

        if self.success and xyz_str:
            return self.get_xyzs(initial_geom=False)[0]
        elif self.success:
            return self.converged_geometries[0]
        else:
            df = self.convergence
            idx = (df[1:] / df.loc['target']).apply(np.linalg.norm, axis=1).idxmin()
            if xyz_str:
                return self.cclib_results.writexyz(indices=idx-1)
            else:
                return self.all_geometries[idx-1]

    def get_irc_midpoint(self,
                         converged=True):
        """
        Find the geometry index of an IRC job where the direction of the path changes.
        """
        scf_e = self.get_scf_energies(converged=converged)
        e_diff = np.diff(scf_e)
        if np.alltrue(e_diff < 0):
            # There is no midpoint
            return
        return np.argmax(e_diff) + 1

    def guess_rxn_from_irc(self,
                           index: int = 0,
                           as_mol_frags: bool = False,
                           inverse: bool = False,
                           backend: str = 'openbabel'):
        """
        Guess the reactants and products from the IRC path. Note: this
        result is not deterministic depending on the pair of conformes you use.
        And this method is only functioning if the job is bidirectional.

        Args:
            index (int): The index of the pair of conformers to use and the it represents
                         the distance from the TS conformer. The larger the far to the end
                         of the IRC path. defaults to 0, the end of each side of the IRC curve.
            as_mol_frags (bool): Whether to return results as mol fragments or as complexes.
                                 Defaults to ``False`` as to return as complexes.
            inverse (bool): Inverse the direction of the reaction. Defaults to ``False``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        if 'irc' not in self.job_type:
            raise RuntimeError('This method is only valid for IRC jobs.')
        midpoint = self.get_irc_midpoint(converged=True)
        if not midpoint:
            return (None, None)
        conv_idxs = self.get_converged_geom_idx(as_numbers=True)
        if (index < 0) or (index > self.num_converged_geoms // 2):
            raise RuntimeError(f'The index should be between 0 and {self.num_converged_geoms // 2}')
        if index == 0:
            idx1, idx2 = conv_idxs[[midpoint - 1, -1]].tolist()
        else:
            idx1, idx2 = conv_idxs[[index, midpoint + index - 1]].tolist()
        r_mol = RDKitMol.FromXYZ(xyz=self.cclib_results.writexyz(indices=idx1), backend=backend)
        p_mol = RDKitMol.FromXYZ(xyz=self.cclib_results.writexyz(indices=idx2), backend=backend)
        r_mol.SaturateMol(multiplicity=self.multiplicity)
        p_mol.SaturateMol(multiplicity=self.multiplicity)

        if inverse:
            r_mol, p_mol = p_mol, r_mol

        if as_mol_frags:
            r_mol, p_mol = r_mol.GetMolFrags(asMols=True), p_mol.GetMolFrags(asMols=True)

        return r_mol, p_mol

    def guess_rxn_from_freq(self,
                            amplitude: float = 1.0,
                            atom_weighted: bool = False,
                            as_mol_frags: bool = False,
                            inverse: bool = False,
                            backend: str = 'openbabel'):
        """
        Guess the reactants and products from the imaginary frequency. Note: this
        result is not deterministic depending on the amplitude you use.

        Args:
            amplitude (int): The amplitude factor on the displacement matrix to generate the
                             guess geometry for the reactant and the product. A smaller factor
                             makes the geometry close to the TS, while a wildly large factor
                             makes the geometry nonphysical.
            atom_weighted (bool): If use the sqrt(atom mass) as a scaling factor to the displacement.
                                  The concern is that light atoms (e.g., H) tend to have larger motions
                                  than heavier atoms.
            as_mol_frags (bool): Whether to return results as mol fragments or as complexes.
                                 Defaults to ``False`` as to return as complexes.
            inverse (bool): Inverse the direction of the reaction. Defaults to ``False``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        if 'freq' not in self.job_type and not self.is_ts:
            raise RuntimeError('This method is only valid for TS frequency jobs.')
        if self.num_neg_freqs != 1:
            raise RuntimeError(f'This may not be a TS, since it has {self.num_neg_freqs}' \
                               f' imaginary frequencies.')

        xyz, disp = self.converged_geometries[0], self.cclib_results.vibdisps[0]
        if atom_weighted:
            atom_weights = np.sqrt(
                self.cclib_results.atommasses[:self.cclib_results.natom].reshape(-1, 1))
        else:
            atom_weights = np.ones((self.cclib_results.natom, 1))
        xyzs = xyz - amplitude * disp * atom_weights, xyz + amplitude * disp * atom_weights
        mol = self.get_mol(backend=backend)
        xyz_strs = []
        for xyz in xyzs:
            mol.SetPositions(xyz)
            xyz_strs.append(mol.ToXYZ())
        r_mol, p_mol = [RDKitMol.FromXYZ(xyz_str, backend=backend) for xyz_str in xyz_strs]
        r_mol.SaturateMol(multiplicity=self.multiplicity)
        p_mol.SaturateMol(multiplicity=self.multiplicity)

        if inverse:
            r_mol, p_mol = p_mol, r_mol

        if as_mol_frags:
            r_mol, p_mol = r_mol.GetMolFrags(asMols=True), p_mol.GetMolFrags(asMols=True)

        return r_mol, p_mol

    ###################################################################
    ####                                                           ####
    ####                         Plot method                       ####
    ####                                                           ####
    ###################################################################

    def plot_convergence(self,
                         logy: bool = True,
                         relative: bool = True,
                         highlight_index: Optional[int] = None,
                         ax=None,
                         ):
        """
        Plot the convergence curve from the convergence table.

        Args:
            logy (bool): If using log scale for the y axis. Defaults to ``True``.
            relative (bool): If plot relative values (to the target values). Defaults to ``True``.
            highlight_index (int): highlight the data corresponding to the given index.
            ax ()
        """
        df = self.convergence
        df = df[1:] / df.loc['target'] if relative else df[1:]
        if ax:
            ax = df.plot(logy=logy, ax=ax)
        else:
            ax = df.plot(logy=logy)
        if relative:
            ax.hlines(y=1,
                      xmin=df.index[0],
                      xmax=df.index[-1],
                      colors='grey', linestyles='dashed',
                      label='ref', alpha=0.5)
        else:
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            for i, item in enumerate(self.opt_criteria):
                ax.hlines(y=self.convergence.loc['target', item],
                          xmin=df.index[0],
                          xmax=df.index[-1],
                          colors=colors[i], linestyles='dashed',
                          alpha=0.5,
                          )
        if highlight_index and highlight_index in df.index:
            df.loc[[highlight_index]].plot(marker='o', ax=ax, legend=False)
        return ax

    def plot_scan_energies(self,
                         converged: bool = True,
                         relative_x: bool = True,
                         relative_y: bool = True,
                         highlight_index: Optional[int] = None,
                         ax: 'matplotlib.pyplot.axes' = None,
                         ):
        """
        Plot the energy curve for the scan trajectory.

        Args:
            converged (bool): If only returns energies for converged geometries. Defaults to ``True``.
            relative_x (bool): If plot relative values (to the initial values).
                               Only valid for `scan` trajectories. Defaults to ``True``.
            relative_y (bool): If plot relative values (to the lowest values). Defaults to ``True``.
            highlight_index (int): highlight the data corresponding to the given index.
            ax (axes): An existing matplotlib axes instance.
        """
        y_params = self.get_scf_energies(converged=True, only_opt=('opt' in self.job_type), relative=True)
        if relative_y:
            baseline_y = 0
        else:
            baseline_y = y_params.min()

        x_params = self.get_scanparams(converged=converged, relative=relative_x)

        ax = ax or plt.axes()
        ax.plot(x_params, y_params)

        xlabel = {2: 'Distance (A)',
                  3: 'Angle (deg)',
                  4: 'Dihedral (deg)'}[len(self.get_scannames(as_list=True)[0])]
        ax.set_xlabel(xlabel)
        ax.set_ylabel('E(SCF) [kcal/mol]')

        # Print reference line
        ax.hlines(y=baseline_y,
                  xmin=x_params.min(),
                  xmax=x_params.max(),
                  colors='grey', linestyles='dashed',
                  label='ref', alpha=0.5)

        if highlight_index and highlight_index < x_params.shape[0]:
            ax.plot(x_params[0], y_params[0], 'ro')
        return ax

    def plot_irc_energies(self,
                          converged: bool = True,
                          relative: bool = True,
                          highlight_index: Optional[int] = None,
                          ax: 'matplotlib.pyplot.axes' = None,
                         ):
        """
        Plot the energy curve for the IRC trajectory. Note, the sequence may be altered if
        the IRC is a two-way calculation.

        Args:
            converged (bool): If only returns energies for converged geometries. Defaults to ``True``.
            relative (bool): If plot relative values (to the highest values). Defaults to ``True``.
            highlight_index (int): highlight the data corresponding to the given index.
            ax (axes): An existing matplotlib axes instance.
        """
        midpoint = self.get_irc_midpoint(converged=converged)
        y_params = self.get_scf_energies(converged=converged)
        if relative:
            y_params -= y_params.max()
        if midpoint:
            y_params[:midpoint] = y_params[midpoint-1::-1]

        x_params = np.arange(1, y_params.shape[0] + 1)

        ax = ax or plt.axes()
        ax.plot(x_params, y_params)
        ax.set(xlabel='Index', ylabel='E(SCF) [kcal/mol]')

        if highlight_index and highlight_index < x_params.shape[0]:
            ax.plot(x_params[0], y_params[0], 'ro')
        return ax

    ###################################################################
    ####                                                           ####
    ####                       Molecule viewer                     ####
    ####                                                           ####
    ###################################################################

    def view_mol(self, backend='openbabel', *args, **kwargs):
        """
        Create a Py3DMol viewer for the molecule. By default, it will shows either
        the initial geometry or the converged geometry depending on what job type
        involved in the file.

        Args:
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        return mol_viewer(self.get_mol(backend=backend), *args, **kwargs)

    def view_freq(self,
                  mode_idx: int = 0,
                  frames: int = 10,
                  amplitude: float = 1.0):
        """
        Create a Py3DMol viewer for the frequency mode.

        mode_idx (int): The index of the frequency mode to display.
        frames (int): The number of frames of the animation. The larger the value,
                      the slower the change of the animation.
        amplitude (float): The magnitude of the mode change.
        """
        xyz = self.get_xyzs(converged=True)[0]
        lines = xyz.splitlines()
        vib_xyz_list = lines[0:2]
        for i, line in enumerate(lines[2:]):
            line = line.strip() + f'{"":12}'+ ''.join([f'{item:<12}' for item in self.cclib_results.vibdisps[mode_idx][i].tolist()])
            vib_xyz_list.append(line)
        vib_xyz = '\n'.join(vib_xyz_list)
        return freq_viewer(vib_xyz, model='xyz', frames=frames, amplitude=amplitude)

    def view_traj(self,
                  align_scan: bool = True,
                  align_frag_idx: int = 1,
                  backend: str = 'openbabel',
                  **kwargs):
        """
        View the trajectory as a Py3DMol animation.

        Args:
            align_scan (bool): If align the molecule to make the animation cleaner.
                               Defaults to ``True``
            align_frag_idx (int): Value should be either 1 or 2. Assign which of the part to be
                                  aligned. Defaults to ``1``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        if 'scan' in self.job_type and align_scan:
            mol = self._process_scan_mol(align_scan=align_scan,
                                         align_frag_idx=align_frag_idx,
                                         backend=backend)
            xyzs = [mol.ToXYZ(confId=i) for i in range(mol.GetNumConformers())]
        elif 'irc' in self.job_type and \
                not (self.schemes['irc'].get('forward') or self.schemes['irc'].get('reverse')):
            mol = self._process_irc_mol(backend=backend)
            xyzs = [mol.ToXYZ(confId=i) for i in range(mol.GetNumConformers())]
        else:
            xyzs = self.get_xyzs(converged=True)
        if len(xyzs) == 1:
            print('Warning: There is only one geomtry in the file.')
        combined_xyzs = ''.join(xyzs)
        return mol_animation(combined_xyzs, 'xyz', **kwargs)

    ###################################################################
    ####                                                           ####
    ####                   Interactive methods                     ####
    ####                                                           ####
    ###################################################################

    def interact_convergence(self, backend: str = 'openbabel'):
        """
        Create a IPython interactive widget to investigate the optimization convergence.

        Args:
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        mol = self.get_mol(converged=False, backend=backend)
        xyzs = self.get_xyzs(converged=False)
        sdfs = [mol.ToMolBlock(confId=i) for i in range(mol.GetNumConformers())]
        def visual(idx):
            mol_viewer(sdfs[idx-1], 'sdf').update()
            ax = plt.axes()
            self.plot_convergence(highlight_index=idx, ax=ax)
            plt.show()
            print(xyzs[idx-1])

        slider = IntSlider(value=0,
                        min=1, max=self.num_all_geoms, step=1,
                        description='Index',
                        )
        if self.convergence.shape[0] - 1 < self.num_all_geoms:
            print('Warning: The last geometry doesn\'t has convergence information due to error.')
        return interact(visual, idx=slider)

    def interact_freq(self):
        """
        Create a IPython interactive widget to investigate the frequency calculation.
        """
        dropdown = Dropdown(
            options=self.freqs,
            value=self.freqs[0],
            description='Freq [cm-1]',
            disabled=False,
        )
        slider1 = IntSlider(value=10, min=1, max=100, step=1, description='Frames',)
        slider2 = FloatLogSlider(value=1.0, min=-1.0, max=1.0, step=0.1, description='Amplitude',)

        def get_freq_viewer(freq, frames, amplitude):
            freq_idx = np.where(self.freqs == freq)[0][0]
            self.view_freq(mode_idx=freq_idx, frames=frames, amplitude=amplitude).update()

        return interact(get_freq_viewer, freq=dropdown,
                        frames=slider1, amplitude=slider2)

    def interact_scan(self,
                      align_scan: bool = True,
                      align_frag_idx: int = 1,
                      backend: str = 'openbabel',
                      **kwargs):
        """
        Create a IPython interactive widget to investigate the scan results.

        Args:
            align_scan (bool): If align the molecule to make the animation cleaner.
                               Defaults to ``True``
            align_frag_idx (int): Value should be either 1 or 2. Assign which of the part to be
                                  aligned. Defaults to ``1``.
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        mol = self._process_scan_mol(align_scan=align_scan,
                                     align_frag_idx=align_frag_idx,
                                     backend=backend)
        sdfs = [mol.ToMolBlock(confId=i) for i in range(mol.GetNumConformers())]
        xyzs = self.get_xyzs(converged=True)

        # Not directly calling plot_scan_energies for better performance
        y_params = self.get_scf_energies(converged=True, only_opt=('opt' in self.job_type), relative=True)
        baseline_y = 0
        x_params = self.get_scanparams(converged=True, relative=True)
        xlabel = {2: 'Distance (A)',
                  3: 'Angle (deg)',
                  4: 'Dihedral (deg)'}[len(self.get_scannames(as_list=True)[0])]
        ylabel = 'E(SCF) [kcal/mol]'

        def visual(idx):
            mol_viewer(sdfs[idx-1], 'sdf').update()
            ax = plt.axes()
            ax.plot(x_params, y_params)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            # Print reference line
            ax.hlines(y=baseline_y,
                  xmin=x_params.min(),
                  xmax=x_params.max(),
                  colors='grey', linestyles='dashed',
                  label='ref', alpha=0.5)
            ax.plot(x_params[idx-1], y_params[idx-1], 'ro')

            plt.show()
            print(xyzs[idx-1])

        slider = IntSlider(value=0,
                        min=1, max=x_params.shape[0], step=1,
                        description='Index',
                        )

        return interact(visual, idx=slider)

    def interact_irc(self, backend: str = 'openbabel'):
        """
        Create a IPython interactive widget to investigate the IRC results.

        Args:
            backend (str): The backend engine for parsing XYZ. Defaults to ``'openbabel'``.
        """
        mol = self._process_irc_mol(backend=backend)
        sdfs = [mol.ToMolBlock(confId=i) for i in range(mol.GetNumConformers())]
        xyzs = self.get_xyzs(converged=True)
        y_params = self.get_scf_energies(converged=True)
        y_params -= y_params.max()
        midpoint = self.get_irc_midpoint()
        if midpoint:
            xyzs = np.array(xyzs)
            xyzs[:midpoint] = xyzs[midpoint-1::-1]
            y_params[:midpoint] = y_params[midpoint-1::-1]
        x_params = np.arange(1, y_params.shape[0]+1)
        xlabel = 'Index'
        ylabel = 'E(SCF) [kcal/mol]'

        def visual(idx):
            mol_viewer(sdfs[idx-1], 'sdf').update()
            ax = plt.axes()
            ax.plot(x_params, y_params)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.plot(x_params[idx-1], y_params[idx-1], 'ro')
            plt.show()
            print(xyzs[idx-1])

        slider = IntSlider(value=0,
                        min=1, max=x_params.shape[0], step=1,
                        description='Index',
                        )

        return interact(visual, idx=slider)


def scheme_to_dict(scheme_str: str) -> dict:
    """
    A function to transform scheme to a dict.

    Args:
        scheme_str (str): the calculation scheme used in a Gaussian job.
    """
    args = re.findall(SCHEME_REGEX, scheme_str)

    schemes = defaultdict(lambda:{})
    for arg in args:
        if arg.startswith('#'):
            continue
        elif 'iop' in arg:
            key, val = arg.split('(')[1].split(')')[0].split('=')
            iop_scheme = schemes['iop']
            iop_scheme[key.strip()] = val.strip()
        elif '=' in arg and '(' in arg:
            scheme_name = arg.split('(')[0].replace('=', '').strip()
            scheme_dict = schemes[scheme_name]
            scheme_vals = arg.split('(')[1].split(')')[0].split(',')
            for val in scheme_vals:
                val = val.strip()
                if '=' not in val:
                    scheme_dict[val] = True
                else:
                    k, v = val.split('=')
                    scheme_dict[k.strip()] = v.strip()
        elif '=' in arg:
            key, val = arg.split('=')
            scheme_dict = schemes[key.strip()]
            scheme_dict[val.strip()] = True
        elif '//' in arg:
            scheme_dict = schemes['LOT']
            sp_lot, opt_lot = arg.split('//')
            scheme_dict['SP'] = sp_lot.strip()
            scheme_dict['OPT'] = opt_lot.strip()
            scheme_dict['LOT'] = arg.strip()
        elif '/' in arg:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in SEMIEMPIRICAL:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in COMPOSITE_METHODS:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in METHODS:
            scheme_dict = schemes['LOT']
            scheme_dict['method'] = arg.strip()
        elif arg in BASIS_SETS:
            scheme_dict['basis_set'] = arg.strip()
        else:
            schemes[arg.strip()] = True

    if 'LOT' not in schemes:
        schemes['LOT'] = {'LOT': f'{DEFAULT_METHOD}/{DEFAULT_BASIS_SET}'}
    elif 'method' in schemes['LOT'] and 'basis_set' not in schemes['LOT']:
        schemes['LOT']['LOT'] = f"{schemes['LOT']['method']}/{DEFAULT_BASIS_SET}"
    elif 'method' not in schemes['LOT'] and 'basis_set' in schemes['LOT']:
        schemes['LOT']['LOT'] = f"{DEFAULT_METHOD}/{schemes['LOT']['basis_set']}"
    if 'freq' in schemes:
        schemes['LOT']['freq'] = schemes['LOT']['LOT']

    return schemes
