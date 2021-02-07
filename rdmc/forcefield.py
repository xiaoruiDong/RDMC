#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module used to deal with force field supported by RDKit.
"""

from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from rdmc.mol import RDKitMol


ROO_TEMPLATE = Chem.MolFromSmarts('[OH0][OH0]')
ROO_TEMPLATE.GetAtoms()[0].SetNumRadicalElectrons(1)
ROO_MOD = {'edit': {'SetAtomicNum': 9,
                    'SetNumRadicalElectrons': 0},
           'remedy': {'SetAtomicNum': 8,
                      'SetNumRadicalElectrons': 1}}


class RDKitFF(object):
    """
    A wrapper to deal with Force field in RDKit.
    """
    def __init__(self, force_field: str = 'mmff94s'):
        """
        Initiate the ``RDKitFF`` by given the name of the force field.

        Args:
            force_field: The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
        """
        self.ForceField = force_field

    @property
    def ForceField(self):
        """
        Return the force field backend.
        """
        return self._ff

    @ForceField.setter
    def ForceField(self, force_field):
        """
        Reset the force field backend.

        Args:
            force_field: The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
        """
        force_field = force_field.lower()
        if not force_field in ['mmff94s', 'mmff94', 'uff']:
            raise ValueError(f'RDKit does not support {force_field}')
        self._ff = force_field

    def IsOptimizable(self, mol: 'Mol'):
        """
        Check if RDKit has the parameters for all atom type in the molecule.

        Args:
            mol (Mol): The molecule to be checked.

        Returns:
            bool: Whether RDKit has parameters.
        """
        mol = mol.ToRWMol() if isinstance(mol, RDKitMol) else mol

        if self._ff in ['mmff94', 'mmff94s']:
            return rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
        else:
            return rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)

    def MakeOptimizable(self,
                        mol: 'RDKitMol',
                        in_place: bool = False):
        """
        Make the molecule able to be optimized by the force field. Known problematic molecules:

        1. RO[O] is not optimizable by MMFF. By changing -O[O] to -OF,
           it allows the geometry to be optimized yielding reasonable results.

        Args:
            mol ('Mol'): Molecule to be changed.
            in_place (bool, optional): Whether to make the change inplace. Defaults to ``False``.

        Returns:
            ('Mol', dict): A modified molecule and approaches to modify this molecule.

        Raises:
            NotImplementedError: Conversion strategy has not been implemented.
        """
        if self._ff == 'uff' or self.IsOptimizable(mol):
            return mol, {}

        if not in_place:
            # Try to make a backup of the molecule if possible
            try:
                mol_copy = mol.Copy()
            except AttributeError:
                mol_copy = mol
        else:
            mol_copy = mol

        edits = {}
        # Convert -O[O] to -OF
        roo = get_roo_radical_atoms(mol_copy)
        for idx in roo:
            atom = mol_copy.GetAtomWithIdx(idx)
            edits[idx] = ROO_MOD
            for attr, value in ROO_MOD['edit'].items():
                getattr(atom, attr)(value)

        # Check if optimizable
        if not self.IsOptimizable(mol_copy):
            raise NotImplementedError('Strategies of making this molecule optimizable has '
                                      'not been implemented.')
        else:
            return mol_copy, edits

    def RecoverMol(self,
                   mol: 'Mol',
                   edits: dict,
                   in_place: bool = True):
        """
        Recover the molecule from modifications.

        Args:
            mol ('Mol'): Molecule to be changed.
            edits (dict): A dict of approach to modify this molecule
            in_place (bool, optional): Whether to make the change inplace. Defaults to ``True``.

        Returns:
            'Mol' A recovered molecule.

        Raises:
            NotImplementedError: Conversion strategy has not been implemented.
        """
        if not in_place:
            # Try to make a backup of the molecule if possible
            try:
                mol_copy = mol.Copy()
            except AttributeError:
                mol_copy = mol
        else:
            mol_copy = mol

        for idx, approach in edits.items():
            atom = mol_copy.GetAtomWithIdx(idx)
            for attr, value in approach['remedy'].items():
                getattr(atom, attr)(value)

        return mol_copy

    def _OptimizeConfs(self,
                  mol: 'Mol',
                  max_iters: int = 200,
                  num_threads: int = 0) -> int:
        """
        A wrapper for force field optimization.

        Args:
            mol ('Mol'): A molecule to be optimized.
            max_iters (int, optional): max iterations. Defaults to ``200``.
            num_threads (int, optional): number of threads to use. Defaults to ``0`` for all.

        Returns:
            - int: 0 for optimization done; 1 for not optimized; -1 for not optimizable.
            - float: energy
        """
        if self._ff == 'mmff94s':
            return rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol,
                                                                 numThreads=num_threads,
                                                                 maxIters=max_iters,
                                                                 mmffVariant='MMFF94s')
        elif self._ff == 'mmff94':
            return rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol,
                                                                 numThreads=num_threads,
                                                                 maxIters=max_iters,
                                                                mmffVariant='MMFF94')
        else:
            return rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol,
                                                                numThreads=num_threads,
                                                                maxIters=max_iters,)

    def OptimizeConfs(self,
                      mol: 'Mol',
                      max_iters: int = 200,
                      num_threads: int = 0,
                      max_cycles: int = 20):
        """
        A wrapper for force field optimization.

        Args:
            mol ('Mol'): A molecule to be optimized.
            max_iters (int, optional): max iterations. Defaults to ``200``.
            num_threads (int, optional): number of threads to use. Defaults to ``0`` for all.
            max_cycles (int, optional): number of outer cycle. Check convergence after maxIters.
                                       Defaults to ``20``.

        Returns:
            - int: 0 for optimization done; 1 for not optimized; -1 for not optimizable.
            - float: energy
        """
        mol = mol.ToRWMol() if isinstance(mol, RDKitMol) else mol

        i = 0
        while i < max_cycles:
            results = self._OptimizeConfs(mol, max_iters, num_threads)
            not_converged = [result[0] for result in results]
            if 1 not in not_converged:
                return results
            i += 1
        else:
            # Hasn't been optimized
            return results


def get_roo_radical_atoms(mol: 'Mol') -> tuple:
    """
    Find the oxygen radical site of peroxide groups in the RDKit molecules.
    This atomtype is currently not optimizable by RDKit built-in MMFF algorithm.

    Args:
        mol: Union[RDKitMol, Mol, RWMol]: RDKitMol or RDKit molecule objects.

    Returns:
        tuple: The atom index of the oxygen radical site.
    """
    roo_matches = mol.GetSubstructMatches(ROO_TEMPLATE)
    O_idx = []
    for match in roo_matches:
        for idx in match:
            if mol.GetAtomWithIdx(idx).GetNumRadicalElectrons() == 1:
                O_idx.append(idx)
    return tuple(set(O_idx))
