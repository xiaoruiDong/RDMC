#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with RDKit Conformer.
"""

from rdkit.Chem.rdchem import Conformer, Mol, RWMol

from rdmc.mol import RDKitMol
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
                mol: Union[RWMol, Mol],
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

    def GetOwningMol(self):
        """
        Get the owning molecule of the conformer.

        Returns:
            Union[Mol, RWMol, RDKitMol]: The owning molecule
        """
        return self._owning_mol

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
                     owning_mol: Union[RDKitMol,
                                       Mol,
                                       RWMol]):
        """
        Set the owning molecule of the conformer. It can be either RDKitMol
        or Chem.rdchem.Mol.

        Args:
            owning_mol: Union[RDKitMol, Chem.rdchem.Mol] The owning molecule of the conformer.

        Raises:
            ValueError: Not a valid ``owning_mol`` input, when giving something else.
        """
        if isinstance(owning_mol, (RDKitMol, Mol, RWMol)):
            self._owning_mol = owning_mol
        else:
            raise ValueError('Not a valid molecule')

