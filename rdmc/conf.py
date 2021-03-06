#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit Conformer.
"""

from typing import Sequence, Union

import numpy as np
from rdkit.Chem import rdMolTransforms as rdMT
from rdkit.Chem.rdchem import Conformer

from rdmc.utils import find_internal_torsions, set_rdconf_coordinates


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
                          indexed1: bool = False):
        """
        Get all of the torsional modes (rotors) of the Conformer. This information
        is obtained from its owning molecule.

        Args:
            indexed1: The atom index in RDKit starts from 0. If you want to have
                       indexed 1 atom indexes, please set this argument to ``True``.

        Returns:
            Optinal[list]: A list of four-atom-indice to indicating the torsional modes.
        """
        try:
            return self._torsions if not not indexed1 \
                else [[ind + 1 for ind in tor] for tor in self._torions]
        except AttributeError:
            self._torsions = find_internal_torsions(self._owning_mol)
            return self._torsions

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
        return rdMT.SetBondLength(self._conf, *atomIds, value)

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
        return rdMT.SetAngleDeg(self._conf, *atomIds, value)

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
        return rdMT.SetAngleRad(self._conf, *atomIds, value)

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
        rdMT.SetDihedralDeg(self._conf, *torsion, degree)

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
