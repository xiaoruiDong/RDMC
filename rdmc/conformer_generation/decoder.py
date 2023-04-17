#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import timer
from rdmc.mol import RDKitMol


class MolDecoder(Task):

    def post_run(self, *args, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        mol = self.last_result
        mol.SetProp("Name",
                    mol.ToSmiles(removeHs=False,
                                 removeAtomMap=False,)
                    )


class SmilesDecoder(MolDecoder):
    """ Decode SMILES to RDKitMol object """
    @timer
    def run(self,
            smiles: str,
            *args,
            **kwargs):
        """
        Decode SMILES to RDKitMol object.

        Args:
            smiles (str): SMILES string
        """
        return RDKitMol.FromSmiles(smiles, *args, **kwargs)


class InchiDecoder(MolDecoder):
    """ Decode InChI to RDKitMol object """
    @timer
    def run(self,
            inchi: str,
            *args,
            **kwargs):
        """
        Decode InChI to RDKitMol object.

        Args:
            inchi (str): InChI string
        """
        return RDKitMol.FromInchi(inchi, *args, **kwargs)


class SmartsDecoder(MolDecoder):
    """ Decode SMARTS to RDKitMol object """
    @timer
    def run(self,
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
    @timer
    def run(self,
            smiles: str,
            *args,
            **kwargs,):
        """
        Decode reaction SMILES to RDKitMol object.

        Args:
            smiles (str): reaction SMILES string
        """
        return [RDKitMol.FromSmiles(smi, *args, **kwargs) for smi in smiles.split(".")]
