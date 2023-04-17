#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task import Task
from rdmc.mol import RDKitMol


class SmilesDecoder(Task):
    """ Decode SMILES to RDKitMol object """
    def task(self,
             smiles: str,
             *args,
             **kwargs):
        """
        Decode SMILES to RDKitMol object.

        Args:
            smiles (str): SMILES string
        """
        return RDKitMol.FromSmiles(smiles, *args, **kwargs)


class InchiDecoder(Task):
    """ Decode InChI to RDKitMol object """
    def task(self,
             inchi: str,
             *args,
             **kwargs):
        """
        Decode InChI to RDKitMol object.

        Args:
            inchi (str): InChI string
        """
        return RDKitMol.FromInchi(inchi, *args, **kwargs)


class SmartsDecoder(Task):
    """ Decode SMARTS to RDKitMol object """

    def task(self,
             smarts: str,
             *args,
             **kwargs):
        """
        Decode SMARTS to RDKitMol object.

        Args:
            smarts (str): SMARTS string
        """
        return RDKitMol.FromSmarts(smarts, *args, **kwargs)


class RxnSmilesDecoder(Task):
    """ Decode reaction SMILES to RDKitMol object """
    def task(self,
             smiles: str,
             *args,
             **kwargs,):
        """
        Decode reaction SMILES to RDKitMol object.

        Args:
            smiles (str): reaction SMILES string
        """
        return [RDKitMol.FromSmiles(smi, *args, **kwargs) for smi in smiles.split(".")]
