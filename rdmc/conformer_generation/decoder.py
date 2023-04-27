#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.task import Task
from rdmc.mol import RDKitMol


class MolDecoder(Task):

    label = 'MolDecoder'

    def post_run(self, **kwargs):
        """
        Set the SMILES as the name of the RDKitMol object.
        """
        self.n_success = 1  # no error raise during run
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
        All child classes should implement this method and has at least `repr`
        as the argument.

        Args:
            reps (str): representation string
        """
        raise NotImplementedError


class SmilesDecoder(MolDecoder):
    """ Decode SMILES to RDKitMol object """

    label = 'SmilesDecoder'

    @MolDecoder.timer
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

    label = 'InchiDecoder'

    @MolDecoder.timer
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

    label = 'SmartsDecoder'

    @MolDecoder.timer
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

    label = 'RxnSmilesDecoder'

    @MolDecoder.timer
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
