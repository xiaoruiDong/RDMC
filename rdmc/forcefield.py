#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module used to deal with force field supported by RDKit.
"""

from typing import Optional, Sequence, Union

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
try:
    # Openbabel 3
    from openbabel import openbabel as ob
except ImportError:
    # Openbabel 2
    import openbabel as ob

from rdmc.mol import RDKitMol
from rdmc.utils import (rdkit_mol_to_openbabel_mol,
                        get_obmol_coords,
                        set_rdconf_coordinates)


ROO_TEMPLATE = Chem.MolFromSmarts('[O;X1]-[O;X2]')
ROO_TEMPLATE.GetAtoms()[0].SetNumRadicalElectrons(1)
ROO_MOD = {'edit': {'SetAtomicNum': 9,
                    'SetNumRadicalElectrons': 0},
           'remedy': {'SetAtomicNum': 8,
                      'SetNumRadicalElectrons': 1}}

H_RAD_TEMP = Chem.MolFromSmiles('[H]')
H_RAD_MOD = {'edit': {'SetAtomicNum': 17,
                      'SetNumRadicalElectrons': 1},
             'remedy': {'SetAtomicNum': 1,
                      'SetNumRadicalElectrons': 1}}

O_RAD_TEMP = Chem.MolFromSmiles('[O]')
O_RAD_MOD = {'edit': {'SetAtomicNum': 9,
                      'SetNumRadicalElectrons': 1},
             'remedy': {'SetAtomicNum': 8,
                        'SetNumRadicalElectrons': 2}}

H2_RAD_TEMP = Chem.MolFromSmiles('[H][H]')
H2_RAD_MOD = {'edit': {'SetAtomicNum': 17,},
              'remedy': {'SetAtomicNum': 1,}}

class RDKitFF(object):
    """
    A wrapper to deal with Force field in RDKit.
    """

    available_force_field = ['mmff94s', 'mmff94', 'uff']

    def __init__(self, force_field: str = 'mmff94s'):
        """
        Initiate the ``RDKitFF`` by given the name of the force field.

        Args:
            force_field: The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
        """
        self.type = force_field

    @property
    def type(self):
        """
        Return the force field backend.
        """
        return self._type

    @type.setter
    def type(self, force_field: str):
        """
        Reset the force field backend.

        Args:
            force_field (str): The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
        """
        force_field = force_field.lower()
        if force_field not in self.available_force_field:
            raise ValueError(f'RDKit does not support {force_field}')
        self._type = force_field

    def add_distance_constraint(self,
                                atoms: Sequence,
                                value: Optional[Union[int, float]] = None,
                                min_len: Optional[Union[int, float]] = None,
                                max_len: Optional[Union[int, float]] = None,
                                relative: bool = False,
                                force_constant: Union[int, float] = 1e5,
                                ):
        """
        Add a distance constraint to the force field. You should set either ``value``
        or ``min_len`` and ``max_len``.

        Args:
            atoms (Sequence): a length-2 sequence indicate the atom index.
            value (int or float): Set distance to a certain value if ``value`` is provided.
            min_len (int or float): The minimum distance value in Angstrom or
                                   in factor of current length (if ``relative==True``).
            max_len (int or float): The maximum distance value in Angstrom or
                                   in factor of current length (if ``relative==True``).
            relative (bool, optional): Whether input as relative distance to the current length.
            force_constant (int or float, optional): the constant in optimization.
        """
        if (value is None) and ((min_len is None) or (max_len is None)):
            raise ValueError('You should set either value or min_len and max_len')
        if len(atoms) != 2:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 2.')
        if not hasattr(self, 'ff'):
            raise ValueError('You need to setup the molecule first to set constraints.')

        atoms = self.update_atom_idx(atoms)
        if self.type == 'uff':
            add_distance_constraint = self.ff.UFFAddDistanceConstraint
        else:
            add_distance_constraint = self.ff.MMFFAddDistanceConstraint

        if value:
            min_len = value
            max_len = value + 0.000001

        add_distance_constraint(*atoms, relative=relative,
                                minLen=min_len, maxLen=max_len,
                                forceConstant=force_constant)

    def add_angle_constraint(self,
                             atoms: Sequence,
                             value: Optional[Union[int, float]] = None,
                             min_angle: Optional[Union[int, float]] = None,
                             max_angle: Optional[Union[int, float]] = None,
                             relative: bool = False,
                             force_constant: Union[int, float] = 1e5,
                             ):
        """
        Add a angle constraint to the force field. You should set either ``value``
        or ``min_angle`` and ``max_angle``.

        Args:
            atoms (Sequence): a length-2 sequence indicate the atom index.
            value (int or float): Set angle to a certain value if ``value`` is provided.
            min_angle (int or float): The minimum angle value in degrees or
                                   in factor of current length (if ``relative==True``).
            max_angle (int or float): The maximum angle value in degrees or
                                   in factor of current length (if ``relative==True``).
            relative (bool, optional): Whether input as relative angle to the current length.
            force_constant (int or float, optional): the constant in optimization.
        """
        if (value is None) and ((min_angle is None) or (max_angle is None)):
            raise ValueError('You should set either value or min_angle and max_angle')
        if len(atoms) != 3:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 3.')
        if not hasattr(self, 'ff'):
            raise ValueError('You need to setup the molecule first to set constraints.')

        atoms = self.update_atom_idx(atoms)
        if self.type == 'uff':
            add_angle_constraint = self.ff.UFFAddAngleConstraint
        else:
            add_angle_constraint = self.ff.MMFFAddAngleConstraint

        if value:
            min_angle = value
            max_angle = value + 0.000001

        add_angle_constraint(*atoms, relative=relative,
                             minAngleDeg=min_angle, maxAngleDeg=max_angle,
                             forceConstant=force_constant)

    def add_torsion_constraint(self,
                               atoms: Sequence,
                               value: Optional[Union[int, float]] = None,
                               min_angle: Optional[Union[int, float]] = None,
                               max_angle: Optional[Union[int, float]] = None,
                               relative: bool = False,
                               force_constant: Union[int, float] = 1e2,
                               ):
        """
        Add a torsion constraint to the force field. You should set either ``value``
        or ``min_angle`` and ``max_angle``.

        Args:
            atoms (Sequence): a length-2 sequence indicate the atom index.
            value (int or float): Set angle to a certain value if ``value`` is provided.
            min_angle (int or float): The minimum angle value in degrees or
                                   in factor of current length (if ``relative==True``).
            max_angle (int or float): The maximum angle value in degrees or
                                   in factor of current length (if ``relative==True``).
            relative (bool, optional): Whether input as relative angle to the current length.
            force_constant (int or float, optional): the constant in optimization.
        """
        if (value is None) and ((min_angle is None) or (max_angle is None)):
            raise ValueError('You should set either value or min_angle and max_angle')
        if len(atoms) != 4:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 4.')
        if not hasattr(self, 'ff'):
            raise ValueError('You need to setup the molecule first to set constraints.')

        atoms = self.update_atom_idx(atoms)
        if self.type == 'uff':
            add_torsion_constraint = self.ff.UFFAddTorsionConstraint
        else:
            add_torsion_constraint = self.ff.MMFFAddTorsionConstraint

        if value:
            min_angle = value
            max_angle = value + 0.000001

        add_torsion_constraint(*atoms, relative=relative,
                               minDihedralDeg=min_angle, maxDihedralDeg=max_angle,
                               forceConstant=force_constant)

    def fix_atom(self,
                 atom_idx: int):
        """
        Fix the coordinates of an atom given by its index.

        Args:
            atom_idx (int): The atom index of the atom to fix.
        """
        atom_idx = self.update_atom_idx([atom_idx])[0]
        self.ff.AddFixedPoint(atom_idx)

    @property
    def mol(self):
        """
        The molecule to be optimized.
        """
        return self._mol

    @mol.setter
    def mol(self,
            mol: Union['Mol', 'RDKitMol']):
        """
        Set the molecule to be optimized.

        Args:
            mol ('Mol', 'RDKitMol'): The molecule to be optimized. Currently,
                                     it supports RDKit Mol and OBMol.
        """
        if isinstance(mol, (Chem.Mol, RDKitMol)):
            self.mol_type = 'rdkit'
            self._mol = mol
        else:
            raise NotImplementedError

    def update_atom_idx(self,
                        atoms: Sequence,
                        ) -> list:
        """
        Update_atom_idx if a rdkit mol atom index is provided.

        Args:
            atoms (Sequence):
        """
        if not hasattr(self, 'mol_type'):
            return atoms

        if self.mol_type == 'rdkit':
            return atoms
        else:
            raise NotImplementedError

    def setup(self,
              mol: Optional[Union['Mol', 'RDKitMol']] = None,
              conf_id: int = -1,
              ignore_interfrag_interactions = False,
              ):
        """
        Setup the force field and get ready to be optimized.

        Args:
            mol (Mol or RDKitMol, optional): Setup the force field based on the molecule.
            conf_id (int, optional): The ID of the conformer to optimize.
            ignore_interfrag_interactions (bool, optional): Whether to ignore interfragment interactions.
                                                            Defaults to ``False`` to allow a similar behavior
                                                            as the other package.
        """
        if mol:
            self.mol = mol
        elif not hasattr(self, 'mol') or not self.mol:
            RuntimeError('You need to set up a molecule to optimize first! '
                         'Either by `RDKitFF.mol = <molecule>`, or '
                         'by `RDKitFF.setup(mol = <molecule>`.')

        if not self.is_optimizable():
            self.mol, self.edits = self.make_optimizable(in_place=True)

        if self.type == 'uff':
            self.ff = rdForceFieldHelpers.UFFGetMoleculeForceField(self.mol.ToRWMol(),
                                                                   confId=conf_id,
                                                                   ignoreInterfragInteractions=ignore_interfrag_interactions)
        else:
            mmff_properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol.ToRWMol(),
                                                                            mmffVariant=self.type,)
            self.ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol.ToRWMol(),
                                                                    pyMMFFMolProperties=mmff_properties,
                                                                    confId=conf_id,
                                                                    ignoreInterfragInteractions=ignore_interfrag_interactions)
        self.ff.Initialize()

    def optimize(self,
                 max_step: int = 100000,
                 tol: float = 1e-8,
                 step_per_iter: int = 5):
        """
        Optimize the RDKit molecule.
        """
        n, success = 0, 1
        while (n < max_step) and (success != 0):
            success = self.ff.Minimize(maxIts=step_per_iter,
                                       energyTol=tol,)
            n = n + step_per_iter

        if success == 0:
            return True

    def is_optimizable(self,
                       mol: Optional['Mol'] = None,
                      ) -> bool:
        """
        Check if RDKit has the parameters for all atom type in the molecule.

        Args:
            mol (Mol): The molecule to be checked.

        Returns:
            bool: Whether RDKit has parameters.
        """
        if not mol:
            mol = self.mol
        mol = mol.ToRWMol() if isinstance(mol, RDKitMol) else mol

        if self.type == 'uff':
            return rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)
        else:
            return rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)

    def make_optimizable(self,
                         mol: Optional['RDKitMol'] = None,
                         in_place: bool = False,
                         ) -> tuple:
        """
        Make the molecule able to be optimized by the force field. Known problematic molecules:

        1. RO[O] is not optimizable by MMFF. By changing -O[O] to -OF,
           it allows the geometry to be optimized yielding reasonable results.
        2. [H].XXX and [O].XXX is not optimizable by MMFF. Changing [H] to [Cl], and [O] to [F].
        3. [H][H] is optimizable by MMFF. Changing to ClCl; when recovery change the bond length
           back to 74 pm.
        Args:
            mol ('Mol'): Molecule to be changed.
            in_place (bool, optional): Whether to make the change inplace. Defaults to ``False``.

        Returns:
            ('Mol', dict): A modified molecule and approaches to modify this molecule.

        Raises:
            NotImplementedError: Conversion strategy has not been implemented.
        """
        mol = mol if mol else self.mol

        if self.is_optimizable(mol):
            return

        if not in_place:
            # Try to make a backup of the molecule if possible
            try:
                mol_copy = mol.Copy()
            except AttributeError:
                # TODO: Print a warning that cannot be done in-place
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

        # Convert [O] to [F] and [H] to [Cl]
        Hs = mol_copy.GetSubstructMatches(H_RAD_TEMP)
        for idx in Hs:
            atom = mol_copy.GetAtomWithIdx(idx[0])
            edits[idx[0]] = H_RAD_MOD
            for attr, value in H_RAD_MOD['edit'].items():
                getattr(atom, attr)(value)

        Os = mol_copy.GetSubstructMatches(O_RAD_TEMP)
        for idx in Os:
            atom = mol_copy.GetAtomWithIdx(idx[0])
            edits[idx[0]] = O_RAD_MOD
            for attr, value in O_RAD_MOD['edit'].items():
                getattr(atom, attr)(value)

        # Convert [H][H] to ClCl
        H2s = [idx for idxs in mol_copy.GetSubstructMatches(H2_RAD_TEMP) for idx in idxs]
        for idx in H2s:
            atom = mol_copy.GetAtomWithIdx(idx)
            edits[idx] = H2_RAD_MOD
            for attr, value in H2_RAD_MOD['edit'].items():
                getattr(atom, attr)(value)

        # Check if optimizable
        if not self.is_optimizable(mol_copy):
            raise NotImplementedError('Strategies of making this molecule optimizable has '
                                      'not been implemented.')
        return mol_copy, edits

    def recover_mol(self,
                   mol: Optional['Mol'] = None,
                   edits: dict = {},
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
        mol = mol if mol else self.mol

        if not in_place:
            # Try to make a backup of the molecule if possible
            try:
                mol_copy = mol.Copy()
            except AttributeError:
                # TODO: Print a warning that cannot be done in-place
                mol_copy = mol
        else:
            mol_copy = mol

        for idx, approach in edits.items():
            atom = mol_copy.GetAtomWithIdx(idx)
            for attr, value in approach['remedy'].items():
                getattr(atom, attr)(value)

        # Adjust H-H bond to 74 pm
        H2s = [idxs for idxs in mol_copy.GetSubstructMatches(H2_RAD_TEMP)]
        for bond in H2s:
            for conformer in mol_copy.GetAllConformers():
                conformer.SetBondLength(bond, 0.74)
        return mol_copy

    def optimize_confs(self,
                       max_step: int = 100000,
                       num_threads: int = 0,
                       step_per_iter: int = 5):
        """
        A wrapper for force field optimization for multiple conformers simultaneously.

        It can take constraints for optimization, but all conformers share the same constraints.
        This can be confusing when using relative value constraints. When using relative constraints,
        the actual value to be used is only based on the first conformer. Therefore, if you are doing
        a torsional scan, you CANNOT first generate conformers with different torsional angles, and
        and then set the relative constraint with value = 1, and optimize using this function. You still
        should iterate conformers and use ``optmize()`` instead.

        Args:
            max_step (int, optional): max iterations. Defaults to ``200``.
            num_threads (int, optional): number of threads to use. Defaults to ``0`` for all.
            step_per_iter (int, optional): number of outer cycle. Check convergence after maxIters.
                                       Defaults to ``20``.

        Returns:
            - int: 0 for optimization done; 1 for not optimized; -1 for not optimizable.
            - float: energy
        """
        if max_step == 0:
            return rdForceFieldHelpers.OptimizeMoleculeConfs(self.mol.ToRWMol(),
                                                             self.ff,
                                                             numThreads=num_threads,
                                                             maxIters=0,)
        i = 0
        while i < max_step:
            results = rdForceFieldHelpers.OptimizeMoleculeConfs(self.mol.ToRWMol(),
                                                                self.ff,
                                                                numThreads=num_threads,
                                                                maxIters=step_per_iter,)
            not_converged = [result[0] for result in results]
            if 1 not in not_converged:
                return results
            i += 1
        try:
            return results
        except UnboundLocalError:
            # It will only happends when users assign a negative value
            # and the variable `results` therefore is not created.
            raise ValueError(f'An invalid max_step {max_step} was assigned. It should be an'
                             f' integer >= 0.')

    def get_optimized_mol(self) -> 'Mol':
        """
        Clean up the optimized molecule and return it. Please call this function once you believe
        the optimization is done, and ready to get the optimized molecule.

        Returns:
            Mol: The optimized RDKitMol molecule
        """
        if hasattr(self, 'edits'):
            self.recover_mol(edits=self.edits)
        return self.mol

    def get_energy(self,):
        """
        Get the energy of the first conformer of the molecule in kcal/mol.

        Returns:
            float: the energy of the index 0 conformer
        """
        return self.ff.CalcEnergy()

    def get_conformer_energies(self,):
        """
        Get the energy of all conformers of the molecule. Surprisingly, there is no such a handy
        function built in the RDKit. The built-in function is CalcEnergy which you can only get the
        energy of the index 0 conformer (the default). The energies are in kcal/mol.

        Returns:
            list: Energies of all conformers of the molecule.
        """
        results = rdForceFieldHelpers.OptimizeMoleculeConfs(self.mol.ToRWMol(),
                                                            self.ff,
                                                            numThreads=0,
                                                            maxIters=0)
        # results is a list of (convergence, energy value)
        return [item[1] for item in results]

    def torsional_scan_1d(self,
                          torsion,
                          num_points: int = 45,
                          rigid: bool = True,
                          init_angle: Optional[float] = None,
                          force_constant: float=100,
                          return_xyz: bool = False):
        mol_copy = self.mol.Copy()
        conf = self.mol.GetConformer()
        if not init_angle:
            init_angle = conf.GetTorsionDeg(torsion)
        pos0 = conf.GetPositions()

        angles = np.zeros(num_points)
        energies = np.zeros_like(angles)
        xyzs = []

        for i in range(num_points):
            conf.SetPositions(pos0)
            angle_val = init_angle + i * 360 / num_points
            conf.SetTorsionDeg(torsion, angle_val)
            self.setup()
            if rigid:
                angles[i] = conf.GetTorsionDeg(torsion)
                energies[i] = self.get_energy()
                if return_xyz:
                    xyzs.append(self.mol.ToXYZ())
                continue
            self.add_torsion_constraint(torsion, value=angle_val,
                                        force_constant=force_constant)
            self.optimize()
            opt_mol = self.get_optimized_mol()
            angles[i] = opt_mol.GetConformer().GetTorsionDeg(torsion)
            energies[i] = self.get_energy()
            if return_xyz:
                xyzs.append(opt_mol.ToXYZ())

        # Initialize the molecule to remove any modifications to self.mol
        self.setup(mol_copy)

        if return_xyz:
            return angles, energies, xyzs
        return angles, energies


class OpenBabelFF:
    """
    A wrapper to deal with Force field of Openbabel. It can handle both RDKit Mol and OBMol
    as input. We suggest to use this class for small scale calculations, due to its slowness.
    """

    available_force_field = ['mmff94s', 'mmff94', 'uff', 'gaff']
    available_solver = ['ConjugateGradients', 'SteepestDescent']

    def __init__(self, force_field: str = 'mmff94s'):
        """
        Initiate the ``OpenbabelFF`` by given the name of the force field.

        Args:
            force_field: The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
                - GAFF
        """
        self.type = force_field
        self.ff = ob.OBForceField.FindForceField(self.type)
        self.solver_type = 'ConjugateGradients'
        self.constraints = None

    @property
    def type(self):
        """
        Return the force field backend.
        """
        return self._type

    @type.setter
    def type(self, force_field: str):
        """
        Reset the force field backend.

        Args:
            force_field (str): The name of the force field. Supporting:
                - MMFF94s (default)
                - MMFF94
                - UFF
        """
        force_field = force_field.lower()
        if force_field not in self.available_force_field:
            raise ValueError(f'Openbabel does not support {force_field}')
        self._type = force_field

    def _initialize_constraints(self):
        """
        Initialize openbabel constraints object.
        """
        if not self.constraints:
            self.constraints = ob.OBFFConstraints()

    def add_distance_constraint(self,
                                atoms: Sequence,
                                value: Union[int, float]):
        """
        Add a distance constraint to the force field.

        Args:
            atoms (Sequence): a length-2 sequence indicate the atom index.
            value (int or float): The distance value in Angstrom.
        """
        if len(atoms) != 2:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 2.')
        self._initialize_constraints()
        atoms = self.update_atom_idx(atoms)
        self.constraints.AddDistanceConstraint(*atoms, value)

    def add_angle_constraint(self,
                             atoms: Sequence,
                             angle: Union[int, float]):
        """
        Add a angle constraint to the force field.

        Args:
            atoms (Sequence): a length-3 sequence indicate the atom index.
            value (int or float): The degree value of the bond angle.
        """
        if len(atoms) != 3:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 3.')
        self._initialize_constraints()
        atoms = self.update_atom_idx(atoms)
        self.constraints.AddAngleConstraint(*atoms, angle)

    def add_torsion_constraint(self,
                               atoms: Sequence,
                               angle: Union[int, float]):
        """
        Add torsion constraint to the force field.

        Args:
            atoms (Sequence): a length-4 sequence indicate the atom index.
            value (int or float): The degree value of the torsion angle.
        """
        if len(atoms) != 4:
            raise ValueError('Invalid `atoms` arguments. Should have a length of 4.')
        self._initialize_constraints()
        atoms = self.update_atom_idx(atoms)
        self.constraints.AddTorsionConstraint(*atoms, angle)

    def fix_atom(self,
                 atom_idx: int):
        """
        Fix the coordinates of an atom given by its index.

        Args:
            atom_idx (int): The atom index of the atom to fix.
        """
        self._initialize_constraints()
        atom_idx = self.update_atom_idx([atom_idx])[0]
        self.constraints.AddAtomConstraint(atom_idx)

    @property
    def mol(self):
        """
        The molecule to be optimized.
        """
        return self._mol

    @mol.setter
    def mol(self,
            mol: Union['Mol', 'RDKitMol', 'OBMol']):
        """
        Set the molecule to be optimized.

        Args:
            mol ('Mol', 'RDKitMol', 'OBMol'): The molecule to be optimized. Currently,
                                              it supports RDKit Mol and OBMol.
        """
        if isinstance(mol, (Chem.Mol, RDKitMol)):
            self.mol_type = 'rdkit'
            self.obmol = rdkit_mol_to_openbabel_mol(mol)
            self._mol = mol
        elif isinstance(mol, ob.OBMol):
            self.mol_type = 'openbabel'
            self.obmol = mol
            self._mol = self.obmol
        else:
            raise NotImplementedError

    def update_atom_idx(self,
                        atoms: Sequence,
                        ) -> list:
        """
        Update_atom_idx if a rdkit mol atom index is provided.

        Args:
            atoms (Sequence):
        """
        if not hasattr(self, 'mol_type'):
            return atoms

        if self.mol_type == 'rdkit':
            return [idx + 1 for idx in atoms]
        elif self.mol_type == 'openbabel':
            return atoms
        else:
            raise NotImplementedError

    def is_optimizable(self,
                       mol: Optional['Mol'] = None,
                      ) -> bool:
        """
        Check if Openbabel has the parameters for all atom type in the molecule.

        Args:
            mol (Mol): The molecule to be checked.

        Returns:
            bool: Whether RDKit has parameters.
        """
        if not mol:
            mol = self.obmol

        # Use an temporary force field to avoid ruinning other parts
        ff = ob.OBForceField.FindForceField(self.type)

        return ff.Setup(mol)

    def setup(self,
              mol: Optional[Union['Mol', 'RDKitMol', 'OBMol']] = None,
              constraints: Optional['ob.OBFFConstraints'] = None,
              ):
        """
        Setup the force field and get ready to be optimized.
        """
        if mol:
            self.mol = mol
        elif not hasattr(self, 'mol') or not self.mol:
            RuntimeError('You need to set up a molecule to optimize first! '
                         'Either by `OpenBabelFF.mol = <molecule>`, or '
                         'by `OpenbabelFF.setup(mol = <molecule>`.')

        if constraints:
            self.constraints = constraints

        if self.constraints:
            return self.ff.Setup(self.obmol, self.constraints)
        else:
            return self.ff.Setup(self.obmol)

    def set_solver(self,
                   solver_type: str):
        if solver_type not in self.available_solver:
            raise ValueError(f'Invalid solver. Got {solver_type}. '
                             f'Should be chose from {self.available_solver}')

    def optimize(self,
                 max_step: int = 100000,
                 tol: float = 1e-8,
                 step_per_iter: int = 5):
        """
        Optimize the openbabel molecule.
        """
        initial_fun = getattr(self.ff, self.solver_type+'Initialize')
        take_n_step_fun = getattr(self.ff, self.solver_type+'TakeNSteps')

        initial_fun(max_step, tol)
        while take_n_step_fun(step_per_iter):
            pass

        self.ff.GetCoordinates(self.obmol)

    def get_optimized_mol(self) -> 'Mol':
        """
        Clean up the optimized molecule and return it. Please call this function once you believe
        the optimization is done, and ready to get the optimized molecule.

        Returns:
            Mol: The optimized molecule
        """
        if self.mol_type == 'rdkit':
            conf = self.mol.GetConformer()
            set_rdconf_coordinates(conf, get_obmol_coords(self.obmol))

        return self.mol


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


def optimize_mol(mol: 'RDKitMol',
                 force_field: str = "MMFF94s",
                 tol: float = 1e-8,
                 step_size: int = 5,
                 max_step: int = 10000,
                 ignore_interfrag_interaction=False,
                 frozen_bonds: list = [],
                 frozen_non_bondings: list = []):
    """
    A helper function to quickly launch forcefield optimization, and
    automatically retry Openbabel force field if RDKit force field fails.
    """
    mol_copy = mol.Copy()
    try:
        # First try if we can use RDKit Forcefield to optimize
        # It is faster, relatively more robust, but have limited atom-type support
        ff = RDKitFF(force_field)
        # In RDKit FF, setup first then add constraints
        ff.setup(mol_copy, ignore_interfrag_interactions= ignore_interfrag_interaction)
        for frozen_bond in frozen_bonds + frozen_non_bondings:
            if len(frozen_bond) == 1:
                ff.add_distance_constraint(atoms=frozen_bond[0], relative=True)
            elif len(frozen_bond) == 2:
                ff.add_distance_constraint(atoms=frozen_bond[0], value=frozen_bond[1])

    except NotImplementedError as e:
        print(e)
        # It usually means we cannot make the molecule optimizable by RDKit force field
        # Then use OpenBabel force field
        ff = OpenBabelFF(force_field)
        # Provides mol can help adjust the atom index difference between RDKit Mol and OpenBabel Mol
        ff.mol = mol_copy
        # In Openbabel Forcefield, add constraints first then setup the force field
        for frozen_bond in frozen_bonds + frozen_non_bondings:
            if len(frozen_bond) == 1:
                distance = np.linalg.norm(mol.GetPositions[frozen_bond, :])
            elif len(frozen_bond) == 2:
                distance = frozen_bond[1]
            ff.add_distance_constraint(frozen_bond[0], value=distance)
        ff.setup()
    finally:
        ff.optimize(max_step=max_step, tol=tol, step_per_iter=step_size)
        mol_copy = ff.get_optimized_mol()
    return mol_copy
