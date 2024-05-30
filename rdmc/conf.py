#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit Conformer.
"""

from typing import Sequence, Union

import numpy as np

from scipy.spatial import distance_matrix

from rdtools.conf import (
    set_conformer_coordinates,
    get_bond_length,
    get_angle_deg,
    get_torsion_deg,
    set_bond_length,
    set_angle_deg,
    set_torsion_deg,
)
from rdtools.dist import has_colliding_atoms
from rdtools.torsion import get_torsional_modes


class EditableConformer(object):
    """
    A wrapper for rdchem.Conformer.

    The method nomenclature follows the Camel style to be consistent with RDKit.
    """

    def __init__(self, conf):
        self._conf = conf
        self._owning_mol = conf.GetOwningMol()
        for attr in dir(conf):
            if not attr.startswith("_") and not hasattr(self, attr):
                setattr(
                    self,
                    attr,
                    getattr(
                        self._conf,
                        attr,
                    ),
                )

    def GetBondLength(
        self,
        atomIds: Sequence[int],
    ) -> float:
        """
        Get the bond length between atoms in Angstrom.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Bond length in Angstrom.
        """
        return get_bond_length(self._conf, atomIds)

    def GetAngleDeg(
        self,
        atomIds: Sequence[int],
    ) -> float:
        """
        Get the angle between atoms in degrees.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Angle value in degrees.
        """
        return get_angle_deg(self._conf, atomIds)

    def GetAngleRad(
        self,
        atomIds: Sequence[int],
    ) -> float:
        """
        Get the angle between atoms in rads.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.

        Returns:
            float: Angle value in rads.
        """
        return self.GetAngleDeg(atomIds) * np.pi / 180.0

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
        return distance_matrix(self.GetPositions(), self.GetPositions())

    def GetOwningMol(self):
        """
        Get the owning molecule of the conformer.

        Returns:
            Union[Mol, RWMol, RDKitMol]: The owning molecule
        """
        return self._owning_mol

    def GetTorsionDeg(
        self,
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
        return get_torsion_deg(self._conf, torsion)

    def GetTorsionalModes(
        self,
        indexed1: bool = False,
        excludeMethyl: bool = False,
        includeRings: bool = False,
    ) -> list:
        """
        Get all of the torsional modes (rotors) of the Conformer. This information
        is obtained from its owning molecule.

        Args:
            indexed1: The atom index in RDKit starts from 0. If you want to have
                       indexed 1 atom indexes, please set this argument to ``True``.
            excludeMethyl (bool): Whether exclude the torsions with methyl groups. Defaults to ``False``.
            includeRings (bool): Whether or not to include ring torsions. Defaults to ``False``.

        Returns:
            Optinal[list]: A list of four-atom-indice to indicating the torsional modes.
        """
        try:
            if not indexed1:
                return self._torsions
            else:
                return [[ind + 1 for ind in tor] for tor in self._torsions]
        except AttributeError:
            self._torsions = get_torsional_modes(
                self._owning_mol,
                exclude_methyl=excludeMethyl,
                include_ring=includeRings,
            )
            return self._torsions

    def HasCollidingAtoms(self, threshold=0.4) -> np.ndarray:
        """
        Whether has atoms are too close together (colliding).

        Args:
            threshold: float indicating the threshold to use in the vdw matrix
        """
        return has_colliding_atoms(
            mol=self._owning_mol,
            conf_id=self.GetId(),
            threshold=threshold,
            reference="vdw",
        )

    def HasOwningMol(self):
        """
        Whether the conformer has a owning molecule.

        Returns:
            bool: ``True`` if the conformer has a owning molecule.
        """
        if self._owning_mol:
            return True
        return False

    def SetOwningMol(self, owningMol: Union["RDKitMol", "Mol", "RWMol"]):
        """
        Set the owning molecule of the conformer. It can be either RDKitMol
        or Chem.rdchem.Mol.

        Args:
            owningMol: Union[RDKitMol, Chem.rdchem.Mol] The owning molecule of the conformer.

        Raises:
            ValueError: Not a valid ``owning_mol`` input, when giving something else.
        """
        if not hasattr(owningMol, "GetConformer"):
            raise ValueError("Provided an invalid molecule object.")
        self._owning_mol = owningMol

    def SetPositions(self, coords: Union[tuple, list]):
        """
        Set the Positions of atoms of the conformer.

        Args:
            coords: a list of tuple of atom coordinates.
        """
        set_conformer_coordinates(self._conf, coords)

    def SetBondLength(
        self,
        atomIds: Sequence[int],
        value: Union[int, float],
    ):
        """
        Set the bond length between atoms in Angstrom.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond length in Angstrom.
        """
        set_bond_length(self._conf, atomIds, value)

    def SetAngleDeg(
        self,
        atomIds: Sequence[int],
        value: Union[int, float],
    ):
        """
        Set the angle between atoms in degrees.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond angle in degrees.
        """
        set_angle_deg(self._conf, atomIds, value)

    def SetAngleRad(
        self,
        atomIds: Sequence[int],
        value: Union[int, float],
    ):
        """
        Set the angle between atoms in rads.

        Args:
            atomIds (Sequence): A 3-element sequence object containing atom indexes.
            value (int or float, optional): Bond angle in rads.
        """
        self.SetAngleDeg(atomIds, value * 180.0 / np.pi)

    def SetTorsionDeg(self, torsion: list, degree: Union[float, int]):
        """
        Set the dihedral angle of the torsion in degrees. The torsion can only be defined
        by a chain of bonded atoms.

        Args:
            torsion (list): A list of four atom indexes.
            degree (float, int): The dihedral angle of the torsion.
        """
        set_torsion_deg(self._conf, torsion, degree)

    def SetAllTorsionsDeg(self, angles: list):
        """
        Set the dihedral angles of all torsional modes (rotors) of the Conformer. The sequence of the
        values are corresponding to the torsions of the molecule (``GetTorsionalModes``).

        Args:
            angles (list): A list of dihedral angles of all torsional modes.
        """
        if len(angles) != len(self.GetTorsionalModes()):
            raise ValueError(
                "The length of angles is not equal to the length of torsional modes"
            )
        for angle, tor in zip(angles, self.GetTorsionalModes()):
            self.SetTorsionDeg(tor, angle)

    def SetTorsionalModes(self, torsions: Union[list, tuple]):
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
            raise ValueError("Invalid torsional mode input.")

    def ToConformer(self) -> "Conformer":
        """
        Get its backend RDKit Conformer object.

        Returns:
            Conformer: The backend conformer
        """
        return self._conf

    def ToMol(self) -> "RDKitMol":
        """
        Convert conformer to mol.

        Returns:
            RDKitMol: The new mol generated from the conformer
        """
        new_mol = self._owning_mol.__class__(self._owning_mol, True)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(self._conf, assignId=True)
        return new_mol
