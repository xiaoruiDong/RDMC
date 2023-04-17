#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task import Task
from rdmc.conformer_generation.utils import timer
from rdmc.mol import RDKitMol


class MolDecoder(Task):

    def post_run(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        mol = self.last_result
        mol.SetProp("Name",
                    mol.ToSmiles(removeHs=False,
                                 removeAtomMap=False,)
                    )

    def run(self,
            *,
            reps: str,
            **kwargs):
        """
        Decode representation (reps) to RDKitMol object.

        Args:
            reps (str): representation string
        """
        raise NotImplementedError


class SmilesDecoder(MolDecoder):
    """ Decode SMILES to RDKitMol object """
    @timer
    def run(self,
            *,
            reps: str,
            **kwargs):
        """
        Decode SMILES to RDKitMol object.

        Args:
            reps (str): SMILES string
        """
        return RDKitMol.FromSmiles(reps, **kwargs)


class InchiDecoder(MolDecoder):
    """ Decode InChI to RDKitMol object """
    @timer
    def run(self,
            *,
            reps: str,
            **kwargs):
        """
        Decode InChI to RDKitMol object.

        Args:
            reps (str): InChI string
        """
        return RDKitMol.FromInchi(reps, **kwargs)


class SmartsDecoder(MolDecoder):
    """ Decode SMARTS to RDKitMol object """
    @timer
    def run(self,
            *,
            reps: str,
            **kwargs):
        """
        Decode SMARTS to RDKitMol object.

        Args:
            reps (str): SMARTS string
        """
        return RDKitMol.FromSmarts(reps, **kwargs)


class RxnSmilesDecoder(Task):
    """ Decode reaction SMILES to RDKitMol object """
    @timer
    def run(self,
            *,
            reps: str,
            **kwargs,):
        """
        Decode reaction SMILES to RDKitMol object.

        Args:
            reps (str): reaction SMILES string
        """
        return [RDKitMol.FromSmiles(smi, **kwargs) for smi in reps.split(".")]
