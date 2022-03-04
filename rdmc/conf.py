#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit Conformer.
"""

from typing import Optional, Sequence, Union

import numpy as np
import scipy.cluster.hierarchy as hcluster

from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT
from rdkit.Chem.rdchem import Conformer
from scipy.spatial import distance_matrix

from rdmc.utils import (find_internal_torsions,
                        find_ring_torsions,
                        set_rdconf_coordinates,
                        VDW_RADII)


class RDKitConf(object):
    """
    A wrapper for rdchem.Conformer.

    The method nomenclature follows the Camel style to be consistent with RDKit.
    """

    def __init__(self, conf):
        self._conf = conf
        self._owning_mol = conf.GetOwningMol()
        for attr in dir(conf):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(self._conf, attr,))

    @classmethod
    def FromConformer(cls,
                      conf: Conformer,
                      ) -> 'RDKitConf':
        """
        Convert a RDKit Chem.rdchem.Conformer to a RDKitConf. This allows a more
        capable and flexible Conformer class.

        Args:
            conf (Chem.rdchem.Conformer): A RDKit Conformer instance to be converted.

        Returns:
            RDKitConf: The conformer corresponding to the RDKit Conformer in RDKitConf
        """
        return cls(conf)

    @classmethod
    def FromMol(cls,
                mol: Union['RWMol', 'Mol'],
                id: int = 0,
                ) -> 'RDkitConf':
        """
        Get a RDKitConf instance from a Chem.rdchem.Mol/RWMol instance.

        Args:
            mol (Union[RWMol, Mol]): a Molecule in RDKit Default format.
            id (int): The id of the conformer to be extracted from the molecule.

        Returns:
            RDKitConf: A Conformer in RDKitConf of the given molecule
        """
        return cls(mol.GetConformer(id))

    @classmethod
    def FromRDKitMol(cls,
                     rdkitmol: 'RDKitMol',
                     id: int = 0,
                     ) -> 'RDkitConf':
        """
        Get a RDKitConf instance from a RDKitMol instance. The owning molecule
        of the generated conformer is RDKitMol instead of Chem.rdchem.Mol.

        Args:
            rdkitmol (RDKitMol): a Molecule in RDKitMol.
            id (int): The id of the conformer to be extracted from the molecule.

        Returns:
            RDKitConf: A Conformer in RDKitConf of the given molecule
        """
        return rdkitmol.GetConformer(id)

    def GetBondLength(self,
                      atomIds: Sequence[int],
                     ) -> float:
        """
        Get the bond length between atoms in Angstrom.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Bond length in Angstrom.
        """
        assert len(atomIds) == 2, ValueError(f'Invalid atomIds. It should be a sequence with a length of 2. Got {atomIds}')
        return rdMT.GetBondLength(self._conf, *atomIds)

    def GetAngleDeg(self,
                    atomIds: Sequence[int],
                    ) -> float:
        """
        Get the angle between atoms in degrees.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Angle value in degrees.
        """
        assert len(atomIds) == 3, ValueError(f'Invalid atomIds. It should be a sequence with a length of 3. Got {atomIds}.')
        return rdMT.GetAngleDeg(self._conf, *atomIds)

    def GetAngleRad(self,
                    atomIds: Sequence[int],
                    ) -> float:
        """
        Get the angle between atoms in rads.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Angle value in rads.
        """
        assert len(atomIds) == 3, ValueError(f'Invalid atomIds. It should be a sequence with a length of 3. Got {atomIds}')
        return rdMT.GetAngleRad(self._conf, *atomIds)

    def GetAllTorsionsDeg(self) -> list:
        """
        Get the dihedral angles of all torsional modes (rotors) of the Conformer. The sequence of the
        values are corresponding to the torsions of the molecule (``GetTorsionalModes``).

        Returns:
            list: A list of dihedral angles of all torsional modes.
        """
        return [self.GetTorsionDeg(tor) for tor in self.GetTorsionalModes()]

    def GetDistanceMatrix(self) -> np.ndarray:
        """
        Get the distance matrix of the conformer.

        Returns:
            array: n x n distance matrix such that n is the number of atom.
        """
        return distance_matrix(self._conf.GetPositions(), self._conf.GetPositions())

    def GetOwningMol(self):
        """
        Get the owning molecule of the conformer.

        Returns:
            Union[Mol, RWMol, RDKitMol]: The owning molecule
        """
        return self._owning_mol

    def GetTorsionDeg(self,
                      torsion: list,
                      ) -> float:
        """
        Get the dihedral angle of the torsion in degrees. The torsion can be defined
        by any atoms in the molecule (not necessarily bonded atoms.)

        Args:
            torsion (list): A list of four atom indexes.

        Returns:
            float: The dihedral angle of the torsion.
        """
        return rdMT.GetDihedralDeg(self._conf, *torsion)

    def GetTorsionalModes(self,
                          indexed1: bool = False,
                          includeRings: bool = False):
        """
        Get all of the torsional modes (rotors) of the Conformer. This information
        is obtained from its owning molecule.

        Args:
            indexed1: The atom index in RDKit starts from 0. If you want to have
                       indexed 1 atom indexes, please set this argument to ``True``.
            includeRings (bool): Whether or not to include ring torsions. Defaults to ``False``.

        Returns:
            Optinal[list]: A list of four-atom-indice to indicating the torsional modes.
        """
        try:
            return self._torsions if not indexed1 \
                else [[ind + 1 for ind in tor] for tor in self._torsions]
        except AttributeError:
            self._torsions = find_internal_torsions(self._owning_mol)
            if includeRings:
                self._torsions += find_ring_torsions(self._owning_mol)
            return self._torsions

    def GetVdwMatrix(self, threshold=0.4) -> Optional[np.ndarray]:
        """
        Get the derived Van der Waals matrix, which can be used to analyze
        the collision of atoms. More information can be found from ``generate_vdw_mat``.

        Args:
            threshold: float indicating the threshold to use in the vdw matrix

        Returns:
            Optional[np.ndarray]: A 2D array of the derived Van der Waals Matrix, if the
                                  the matrix exists, otherwise ``None``.
        """
        try:
            return self._vdw_mat
        except AttributeError:
            # Try to obtain from its Owning molecule
            self._vdw_mat = self._owning_mol.GetVdwMatrix(threshold=threshold)
            return self._vdw_mat

    def HasCollidingAtoms(self, threshold=0.4) -> np.ndarray:
        """
        Args:
            threshold: float indicating the threshold to use in the vdw matrix
        """
        dist_mat = np.triu(self.GetDistanceMatrix())
        # if the distance is smaller than a threshold, the atom has a high chance of colliding
        return not np.all(self.GetVdwMatrix(threshold=threshold) <= dist_mat)

    def HasOwningMol(self):
        """
        Whether the conformer has a owning molecule.

        Returns:
            bool: ``True`` if the conformer has a owning molecule.
        """
        if self._owning_mol:
            return True
        return False

    def SetOwningMol(self,
                     owningMol: Union['RDKitMol',
                                      'Mol',
                                      'RWMol']):
        """
        Set the owning molecule of the conformer. It can be either RDKitMol
        or Chem.rdchem.Mol.

        Args:
            owningMol: Union[RDKitMol, Chem.rdchem.Mol] The owning molecule of the conformer.

        Raises:
            ValueError: Not a valid ``owning_mol`` input, when giving something else.
        """
        if not hasattr(owningMol, 'GetConformer'):
            raise ValueError('Provided an invalid molecule object.')
        self._owning_mol = owningMol

    def SetPositions(self,
                     coords: Union[tuple, list]):
        """
        Set the Positions of atoms of the conformer.

        Args:
            coords: a list of tuple of atom coordinates.
        """
        set_rdconf_coordinates(self._conf, coords)

    def SetBondLength(self,
                      atomIds: Sequence[int],
                      value: Union[int, float],
                     ) -> float:
        """
        Set the bond length between atoms in Angstrom.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond length in Angstrom.
        """
        assert len(atomIds) == 2, ValueError(f'Invalid atomIds. It should be a sequence with a length of 2. Got {atomIds}')
        try:
            return rdMT.SetBondLength(self._conf, *atomIds, value)
        except ValueError:
            # RDKit doesn't allow change bonds for non-bonding atoms
            # A workaround may be form a bond and change the distance
            try:
                edit_conf_by_add_bonds(self, 'SetBondLength', atomIds, value)
            except ValueError:
                # RDKit doesn't allow change bonds for atoms in a ring
                # A workaround hasn't been proposed
                raise NotImplementedError(f'Approach for modifying the bond length of {atomIds} is not available.')

    def SetAngleDeg(self,
                    atomIds: Sequence[int],
                    value: Union[int, float],
                    ) -> float:
        """
        Set the angle between atoms in degrees.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond angle in degrees.
        """
        assert len(atomIds) == 3, ValueError(f'Invalid atomIds. It should be a sequence with a length of 3. Got {atomIds}.')
        try:
            return rdMT.SetAngleDeg(self._conf, *atomIds, value)
        except ValueError:
            try:
                # RDKit doesn't allow change bonds for non-bonding atoms
                # A workaround may be form a bond and change the distance
                edit_conf_by_add_bonds(self, 'SetAngleDeg', atomIds, value)
            except:
                # RDKit doesn't allow change bonds for atoms in a ring
                # A workaround hasn't been proposed
                raise NotImplementedError(f'Approach for modifying the bond angle of {atomIds} is not available.')

    def SetAngleRad(self,
                    atomIds: Sequence[int],
                    value: Union[int, float],
                    ) -> float:
        """
        Set the angle between atoms in rads.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond angle in rads.
        """
        assert len(atomIds) == 3, ValueError(f'Invalid atomIds. It should be a sequence with a length of 3. Got {atomIds}')
        try:
            return rdMT.SetAngleRad(self._conf, *atomIds, value)
        except ValueError:
            try:
                # RDKit doesn't allow change bonds for non-bonding atoms
                # A workaround may be form a bond and change the distance
                edit_conf_by_add_bonds(self, 'SetAngleRad', atomIds, value)
            except:
                # RDKit doesn't allow change bonds for atoms in a ring
                # A workaround hasn't been proposed
                raise NotImplementedError(f'Approach for modifying the bond angle of {atomIds} is not available.')

    def SetTorsionDeg(self,
                      torsion: list,
                      degree: Union[float, int]):
        """
        Set the dihedral angle of the torsion in degrees. The torsion can only be defined
        by a chain of bonded atoms.

        Args:
            torsion (list): A list of four atom indexes.
            degree (float, int): The dihedral angle of the torsion.
        """
        try:
            rdMT.SetDihedralDeg(self._conf, *torsion, degree)
        except ValueError:
            try:
                # RDKit doesn't allow change bonds for non-bonding atoms
                # A workaround may be form a bond and change the distance
                edit_conf_by_add_bonds(self, 'SetDihedralDeg', torsion, degree)
            except:
                # RDKit doesn't allow change bonds for atoms in a ring
                # A workaround hasn't been proposed
                raise NotImplementedError(f'Approach for modifying the dihedral of {torsion} is not available.')

    def SetAllTorsionsDeg(self, angles: list):
        """
        Set the dihedral angles of all torsional modes (rotors) of the Conformer. The sequence of the
        values are corresponding to the torsions of the molecule (``GetTorsionalModes``).

        Args:
            angles (list): A list of dihedral angles of all torsional modes.
        """
        if len(angles) != len(self.GetTorsionalModes()):
            raise ValueError(
                'The length of angles is not equal to the length of torsional modes')
        for angle, tor in zip(angles, self.GetTorsionalModes()):
            self.SetTorsionDeg(tor, angle)

    def SetTorsionalModes(self,
                          torsions: Union[list, tuple]):
        """
        Set the torsional modes (rotors) of the Conformer. This is useful when the
        default torsion is not correct.

        Args:
            torsions (Union[list, tuple]): A list of four-atom-lists indicating the torsional modes.

        Raises:
            ValueError: The torsional mode used is not valid.
        """
        if isinstance(torsions, (list, tuple)):
            self._torsions = torsions
        else:
            raise ValueError('Invalid torsional mode input.')

    def ToConformer(self) -> 'Conformer':
        """
        Get its backend RDKit Conformer object.

        Returns:
            Conformer: The backend conformer
        """
        return self._conf


class ConformerCluster(object):

    def __init__(self,
                 children: 'np.array',
                 energies: 'np.array' = None):
        """
        A Class for storing conformer cluster information. The head and energy attributes store
        the representative conformer's index and its corresponding energies. And children and energies
        attributes store all of the conformer indexes and energies within this cluster. There is not limit to
        the cluster definition. It can be a set of conformers with almost identical geometries, or a completely
        set of random geometries.
        """
        self.children = children
        if energies is None:
            self.energies = np.zeros_like(self.children)
        else:
            self.energies = energies
        self._update_energy_and_head()

    def __repr__(self,):
        "Text when printing out the object"
        return f'Conformer(head={self.head:d}, energy={self.energy:.5f}, n_children={len(self.children)})'

    def split_by_energies(self,
                          decimals: int = 1,
                          as_dict: bool = True):
        """
        Split the conformer by energies.

        Args:
            decimals (int, optional): clustering children based on the number of digit
                                      after the dot of the energy values.
                                      For kcal/mol and J/mol, 0 or 1 is recommended; for 
                                      hartree, 3 or 4 is recommended. Defaults to ``1``.
            as_dict (bool, optional): If ``True``, return a dict object whose keys are
                                      energy values and values are divided ConformerCluster
                                      object. Otherwise, return a list of ConformerClusters.
                                      Defaults to ``True``.
        """
        # Get the energy levels according to the provided decimal accuracy
        rounded_e = np.round(self.energies, decimals=decimals)
        energy_levels = np.unique(rounded_e)

        # Create new clusters based on the energy division
        n_clusters = {}
        for e_level in energy_levels:
            indexes = rounded_e == e_level
            n_clusters[e_level] = ConformerCluster(children=self.children[indexes],
                                                   energies=self.energies[indexes])
        if as_dict:
            return n_clusters
        else:
            return list(n_clusters.values())

    def merge(self,
              clusters: list,):
        """
        Merge the cluster with a list of clusters.

        Args:
            clusters (list): A list of the ConformerCluster object to be merged in.
        """
        # Create a list with all clusters
        new_cluster_list = [self] + clusters

        # Update the head and energy for the current cluster
        min_energy_idx = np.argmin([c.energy for c in new_cluster_list])
        self.head, self.energy, = (new_cluster_list[min_energy_idx].head,
                                   new_cluster_list[min_energy_idx].energy)

        # Update the children and energies for the current cluster
        self.children, self.energies = (np.concatenate([c.children for c in new_cluster_list]),
                                       np.concatenate([c.energies for c in new_cluster_list]))

    def _update_energy_and_head(self):
        """
        Update the attributes of head and energy when creating the cluster
        based on the ones with the minimum energy.
        """
        idx = np.argmin(self.energies)
        self.head = self.children[idx]
        self.energy = self.energies[idx]


class ConformerFilter(object):

    def __init__(self,
                 mol: 'RDKitMol',):
        """
        The Filtration Class for filtering conformer clusters.
        It is designed to be used with the ConformerCluster object.
        """
        self.mol = mol

    def get_torsional_angles(self,
                             confid: int,
                             adjust_periodicity: bool = True,
                            ) -> 'np.array':
        """
        Get torsional angles for a given conformers.

        Args:
            confid (int): The conformer ID.
            adjust_periodicity (bool): Whether to adjust the periodicity for torsional angles.
                                       Defaults to ``True``.

        Returns:
            np.array: A 1D array of torsional angles.
        """
        conf = self.mol.GetConformer(id=confid)
        tor_angle = conf.GetAllTorsionsDeg()
        if adjust_periodicity:
            for i in range(len(tor_angle)):
                tor_angle[i] += 360 if tor_angle[i] < -170 else 0
        return np.array(tor_angle)

    def get_tor_matrix(self,
                       confs: Union['np.array',list],
                       adjust_periodicity: bool = True,
                      ) -> np.ndarray:
        """
        Get the torsional matrix consists of torsional angles for the input list of conformer indexes.

        Args:
            confid (int): The conformer ID
            adjust_periodicity (bool): Whether to adjust the periodicity for torsional angles.

        Returns:
            np.array: a M (the number of conformers indicated by confs) x N (the number of torsional angles) matrix
        """
        tor_matrix = []
        for conf in confs:
            tor_matrix.append(self.get_torsional_angles(int(conf),
                                                        adjust_periodicity=adjust_periodicity,
                                                        ))
        return np.array(tor_matrix)

    def _get_results_from_cluster_idxs(self,
                                       confs: Union[list,'np.array'],
                                       c_idxs: Union[list,'np.array'],
                                       as_dict: bool,
                                       as_list_idx: bool,):
        """
        A helper function to convert a list of cluster indexes into desired format.
        """
        clusters = {idx: [] for idx in np.unique(c_idxs)}
        for list_idx, c_idx in enumerate(c_idxs):
            element = list_idx if as_list_idx else confs[list_idx]
            clusters[c_idx].append(element)
        if as_dict:
            return clusters
        else:
            return list(clusters.values())

    def _get_clusters_from_grouped_idxs(self,
                                        old_clusters: list,
                                        grouped_conf_idx: list,
                                        ):
        """
        A helper function to convert a list of original clusters and a list of grouped conformer
        indexes to new clusters.
        """
        new_clusters = []
        for conf_idx in grouped_conf_idx:
            new_cluster = old_clusters[conf_idx[0]]
            new_cluster.merge([old_clusters[j] for j in conf_idx[1:]])
            new_clusters.append(new_cluster)
        return new_clusters

    def hierarchy_cluster(self,
                          confs,
                          threshold: float = 5.,
                          criterion: str = 'distance',
                          method: str = 'average',
                          adjust_periodicity: bool = False,
                          as_dict: bool = True,
                          as_list_idx: bool = False,
                         ):
        """
        The implementation of an hierarchy clustering method based on scipy.
        It is basically defining clusters based on points within a hypercube defined by threhold.
        More details refer to:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html

        Args:
            confs (list): A list of conformer IDs.
            threshold (float, optional): The threshold (in degree) used for hierarchy clustering. Defaults to 5.
            criterion (str, optional): Specifies the criterion for forming flat clusters. Valid values are ‘inconsistent’,
                                      ‘distance’ (default), or ‘maxclust’ cluster formation algorithms
            method (str, optional): The linkage method to use. Valid values are single, complete, average (default),
                                    weighted, median centroid, and ward. Except median centroid (O(n^3)), others have a
                                    computational cost scaled by O(n^2).
            adjust_periodicity (bool, optional): Since dihedral angles have a period of 360 degrees. Defaults to ``True``.
                                                 It is suggested to run twice with this value be ``True`` and ``False`` to
                                                 get a better performance.
            as_dict (bool): Return the result as a dict object with keys for the index of clusters and values
                            of conformer indexes (provided in confs). Otherwise, return as a list of grouped
                            conformer indexes. Defaults to ``True``.
            as_list_idx (bool): Return the indexes in the `confs` list other than the value in the ``confs``.
                                Default to ``False``.
        """
        # Generate torsional matrix
        tor_matrix = self.get_tor_matrix(confs, adjust_periodicity=adjust_periodicity)

        # Calculate the clusters
        # The output is an array of the same size as 'confs', but with numbers indicating the belonging
        # cluster indexes. E.g., [1,1,1,2,2,2]
        c_idxs = hcluster.fclusterdata(tor_matrix,
                                         threshold,
                                         criterion=criterion,
                                         method=method)

        return self._get_results_from_cluster_idxs(confs=confs, c_idxs=c_idxs,
                                                   as_dict=as_dict, as_list_idx=as_list_idx)

    def filter_by_iter_hcluster(self,
                                clusters: Union[list, 'ConformerCluster'],
                                threshold: float = 5.,
                                criterion: str = 'distance',
                                method: str = 'average',
                                max_iter: int = 10,
                                as_clusters: bool = True,):
        """
        A method to filter comformers by iteratively applying hierarchy clustering.
        In a new iteration, only the distinguishable representative conformers will be used that are
        generated in the previous iteration.

        Args:
            clusters (ConformerCluster or list): A single ConformerCluster object or a list of conformerClusters.
            threshold (float, optional): The threshold (in degree) used for hierarchy clustering. Defaults to 5.
            criterion (str, optional): Specifies the criterion for forming flat clusters. Valid values are ‘inconsistent’,
                                      ‘distance’ (default), or ‘maxclust’ cluster formation algorithms
            method (str, optional): The linkage method to use. Valid values are single, complete, average (default),
                                    weighted, median centroid, and ward. Except median centroid (O(n^3)), others have a
                                    computational cost scaled by O(n^2).
            adjust_periodicity (bool, optional): Since dihedral angles have a period of 360 degrees. Defaults to ``True``.
                                                 It is suggested to run twice with this value be ``True`` and ``False`` to
                                                 get a better performance.
            max_iter (int, optional): The max number of iterations. Defaults to 10. There is a early-stopping techinque
                                      if number of clusters doesn't change with increasing number of iterations.
            as_clusters (bool, optional): Return the results as a list of ConformerClusters (``True``) or a list of
                                          grouped conformer indexes (``True``). Defaults to ``True``.

        Return:
            list
        """
        # If the input `clusters` is a single ConformerCluster
        # This is often the case if the cluster is generated from energy clustering.
        # Then create the list of clusters based on its children.
        if isinstance(clusters, ConformerCluster):
            c_tmp = self.hierarchy_cluster(clusters.children,
                                           threshold=threshold,
                                           criterion=criterion,
                                           method=method,
                                           as_dict=False,
                                           as_list_idx=True,
                                           adjust_periodicity=True)
            clusters = [ConformerCluster(clusters.children[c],
                                         clusters.energies[c]) for c in c_tmp]
            max_iter -= 1

        last_num_clusters = len(clusters)
        for i in range(max_iter):
            # Get the representative conformer from each cluster
            # And use them to do further hierarchy clustering
            confs = [cl.head for cl in clusters]
            grouped_conf_idx = self.hierarchy_cluster(confs,
                                                      threshold=threshold,
                                                      criterion=criterion,
                                                      method=method,
                                                      as_dict=False,
                                                      as_list_idx=True,
                                                      adjust_periodicity=i%2)
            clusters = self._get_clusters_from_grouped_idxs(clusters, grouped_conf_idx)
            cur_num_clusters = len(clusters)
            if cur_num_clusters == last_num_clusters:
                break
            last_num_clusters = cur_num_clusters

        if as_clusters:
            return clusters
        else:
            return [c.children for c in clusters]

    def check_dihed_angle_diff(self,
                               confs,
                               threshold: float = 5.0,
                               mask: Optional['np.ndarray'] = None,
                               adjust_periodicity: bool = False,
                               as_dict: bool = True,
                               as_list_idx: bool = False,
                               ):
        """
        The implementation of checking individual dihedral angle difference. This approach is also
        implemented in the MSTor workflow.

        Args:
            confs (list): A list of conformer IDs.
            threshold (float, optional): The difference (in degree) used for filtering. Defaults to 5.
            mask (np.ndarray, optional): An array that has the same length as torsions and values of True or False to
                                         indicate which torsions are not considered. This can be helpful, e.g., to exclude
                                         methyl torsional symmetry.
            adjust_periodicity (bool, optional): Since dihedral angles have a period of 360 degrees. Defaults to ``True``.
                                                 It is suggested to run twice with this value be ``True`` and ``False`` to
                                                 get a better performance.
            as_dict (bool): Return the result as a dict object with keys for the index of clusters and values
                            of conformer indexes (provided in confs). Otherwise, return as a list of grouped
                            conformer indexes. Defaults to ``True``.
            as_list_idx (bool): Return the indexes in the `confs` list other than the value in the ``confs``.

        Returns:
            dict or list
        """
        len_confs = confs.shape[0] if isinstance(confs, np.ndarray) else len(confs)

        if mask:
            # Allowing mask some dimensions
            masked_tor = lambda tor: np.ma.masked_array(tor, mask)
        else:
            masked_tor = lambda tor: tor

        # Create an array to store cluster indexes
        # Initializing all elements to -1
        c_idxs = np.full(len_confs, -1)

        cur_c_idx = 0
        for i in range(len_confs):
            if c_idxs[i] != -1:
                # Assigned previously
                continue
            # Assign unassigned conf to a new cluster
            c_idxs[i] = cur_c_idx
            # Get the torsional angles
            tor_1 = self.get_torsional_angles(confid=int(confs[i]),
                                              adjust_periodicity=adjust_periodicity)
            for j in range(i + 1, len_confs):
                if c_idxs[j] != -1:
                    # Assigned previously
                    continue
                # Get the torsional angles
                tor_2 = self.get_torsional_angles(confid=int(confs[j]),
                                                  adjust_periodicity=adjust_periodicity)
                if np.all(np.abs(masked_tor(tor_1 - tor_2)) < threshold):
                    # Difference of all of the dimensions is smaller than threshold
                    c_idxs[j] = cur_c_idx
            cur_c_idx += 1

        return self._get_results_from_cluster_idxs(confs=confs, c_idxs=c_idxs,
                                                   as_dict=as_dict, as_list_idx=as_list_idx)

    def filter_by_dihed_angles(self,
                               clusters,
                               threshold: float = 5.,
                               mask: 'np.ndarray' = None,
                               as_clusters: bool = True,
                               ) -> list:
        """
        A method to filter comformers by calculating the differences of dihedral angles between conformers.
        The check will be implemented twice, with and without considering periodicity.

        Args:
            clusters (ConformerCluster or list): A single ConformerCluster object or a list of conformerClusters.
            threshold (float, optional): The threshold (in degree) used for hierarchy clustering. Defaults to 5.
            mask (np.array, optional): An array that has the same length as torsions and values of True or False to
                                       indicate which torsions are not considered. This can be helpful, e.g., to exclude
                                       methyl torsional symmetry.
            as_clusters (bool, optional): Return the results as a list of ConformerClusters (``True``) or a list of
                                          grouped conformer indexes (``True``). Defaults to ``True``.

        Return:
            list
        """
        max_iter = 2
        # If the input `clusters` is a single ConformerCluster
        # This is often the case if the cluster is generated from energy clustering.
        # Then create the list of clusters based on its children.
        if isinstance(clusters, ConformerCluster):
            c_tmp = self.check_dihed_angle_diff(clusters.children,
                                                threshold=threshold,
                                                mask=mask,
                                                as_dict=False,
                                                as_list_idx=True,
                                                adjust_periodicity=True)
            clusters = [ConformerCluster(clusters.children[c],
                                         clusters.energies[c]) for c in c_tmp]
            max_iter-=1

        for i in range(max_iter):
            confs = [cl.head for cl in clusters]
            grouped_conf_idx = self.check_dihed_angle_diff(confs,
                                                threshold=threshold,
                                                mask=mask,
                                                as_dict=False,
                                                as_list_idx=True,
                                                adjust_periodicity=i%2)

            clusters = self._get_clusters_from_grouped_idxs(clusters, grouped_conf_idx)

        if as_clusters:
            return clusters
        else:
            return [c.children for c in clusters]

    @property
    def atom_maps(self):
        """
        Store all possible atommappings of the given molecules. There are usually combinatory
        explosion.
        """
        try:
            return self._atom_maps
        except AttributeError:
            self.reset_atom_maps(max_atom_maps=100000)
            return self._atom_maps

    def reset_atom_maps(self, max_atom_maps=100000):
        """
        Reset the stored matches.

        Args:
            max_atom_maps (int): The maximum number of atom maps to generate To avoid combinatory explosion,
                                 it is set to avoid the program to run forever. As a cost, you may miss some mappings.
        """
        matches = self.mol.GetSubstructMatches(self.mol,
                                               uniquify=False,
                                               maxMatches=max_atom_maps)
        self._atom_maps = [list(enumerate(match)) for match in matches]
        if len(self._atom_maps) == 100000:
                print('WARNING: The atom index mappings are not complete (possibly due to the'
                      'large size of the molecule or high symmetry). You may want to regenerate'
                      'atom mappings by `reset_atom_maps` with a number larger than 100000.')


    def pairwise_rmsd(self,
                      i: int,
                      j: int,
                      reflect: bool = False,
                      reorder: bool = True,
                      max_iters: int = 1000,
                      ):
        """
        The implementation of calculating pairwise RMSD values.

        Args:
            i (int): Index of one of the conformer. Usually, used as the 'reference'.
            j (int): Index of the other conformer.
            reflect (bool, optional): Whether to reflect the j conformer to rule out 
                                      mirror symmetry. Defaults to ``False``.
            reorder (bool, optional): Whether to allow atom index order to change (based
                                      on isomorphism check to rule out torsional symmetry
                                      and rotational symmetry).
            max_iters (int, optional): The max iteration in mimizing the RMSD.
        """
        atom_maps = self.atom_maps if reorder else self.atom_maps[:1]

        return self.mol.AlignMol(self.mol,
                                 prbCid=j,
                                 refCid=i,
                                 atomMaps=atom_maps,
                                 reflect=reflect,
                                 maxIters=max_iters
                                 )

    def generate_rmsds_of_cluster(self,
                                  cluster,
                                  reflect: bool = False,
                                  reorder: bool = True,
                                  max_iters: int = 1000,):
        """
        Get the RMSDs between the representative conformer and each conformer in the cluster.
        """
        ref_id = cluster.head
        rmsds = np.zeros(len(cluster.children))
        for i, prb_id in enumerate(cluster.children):
            rmsds[i] = self.pairwise_rmsd(int(ref_id), int(prb_id),
                                          reflect=reflect,
                                          reorder=reorder,
                                          max_iters=max_iters
                                          )
        return rmsds


def edit_conf_by_add_bonds(conf, function_name, atoms, value):
    """
    RDKit forbids modifying internal coordinates with non-bonding atoms.
    This function tries to provide a workaround.

    Args:
        conf (RDKitConf): The conformer to be modified.
        function_name (str): The function name of the edit.
        atoms (list): A list of atoms representing the internal coordinates.
        value (float): Value to be set.
    """
    tmp_mol = conf.GetOwningMol()
    all_bonds = tmp_mol.GetBondsAsTuples()
    tmp_atoms = sorted(atoms)
    bonds_to_add = []
    for i in range(len(tmp_atoms) - 1):
        if not (tmp_atoms[i], tmp_atoms[i+1]) in all_bonds:
            bonds_to_add.append([tmp_atoms[i], tmp_atoms[i+1]])
    tmp_mol = tmp_mol.AddRedundantBonds(bonds_to_add)
    tmp_mol.SetPositions(conf.GetPositions())
    tmp_conf = tmp_mol.GetConformer()
    getattr(rdMT, function_name)(tmp_conf._conf, *atoms, value)
    conf.SetPositions(tmp_conf.GetPositions())
