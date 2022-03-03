#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import logging
import unittest

import numpy as np

from rdkit import Chem
try:
    from openbabel import pybel
except ImportError:
    import pybel

from rdmc import RDKitMol

logging.basicConfig(level=logging.DEBUG)

################################################################################

class TestRDKitMol(unittest.TestCase):
    """
    A class used to test basic operations for the RDKitMol Class.
    """
    smi1 = "CC"

    def test_generate_mol_from_rdkit_rwmol(self):
        """
        Test generating the RDKitMol object from a RDKit RWMol object.
        """
        # Generate a rdkit `RWMol` object
        # The RDKitMol class will set this object as the 'backend' molecule
        rdkit_mol = Chem.RWMol(Chem.rdmolops.AddHs(Chem.MolFromSmiles(self.smi1)))

        # Option 1
        # Initiate the object with rdkit_mol as an argument
        mol = RDKitMol(rdkit_mol, keepAtomMap=False)

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(mol._mol, rdkit_mol)
        self.assertEqual(mol.ToRWMol(), rdkit_mol)

        # Option 2
        mol = RDKitMol.FromMol(rdkit_mol, keepAtomMap=False)
        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(mol._mol, rdkit_mol)
        self.assertEqual(mol.ToRWMol(), rdkit_mol)

    def test_methods_inherent_from_rdkit_mol(self):
        """
        Test if methods are successfully inherent from RDKit Mol object
        """
        # Generate an rdkit `RWMol` object and an RDKitMol
        # The RDKitMol class will set this object as the 'backend' molecule
        rdkit_mol = Chem.RWMol(Chem.rdmolops.AddHs(Chem.MolFromSmiles(self.smi1)))
        mol = RDKitMol(rdkit_mol, keepAtomMap=False)

        # A list of methods of RDKit RWMol objects is given
        # This list may need update when RDKit updates.
        # There are minor differences among versions.
        # Commented are the ones modified in the RDKitMol
        list_of_methods = [
            'AddAtom',
            'AddBond',
            'AddConformer',
            'BeginBatchEdit',
            'ClearComputedProps',
            'ClearProp',
            'CommitBatchEdit',
            'Compute2DCoords',
            'ComputeGasteigerCharges',
            'Debug',
            'GetAromaticAtoms',
            'GetAtomWithIdx',
            'GetAtoms',
            'GetAtomsMatchingQuery',
            'GetBondBetweenAtoms',
            'GetBondWithIdx',
            'GetBonds',
            'GetBoolProp',
            # 'GetConformer',
            # 'GetConformers',
            'GetDoubleProp',
            'GetIntProp',
            'GetMol',
            'GetNumAtoms',
            'GetNumBonds',
            'GetNumConformers',
            'GetNumHeavyAtoms',
            'GetProp',
            'GetPropNames',
            'GetPropsAsDict',
            'GetRingInfo',
            'GetStereoGroups',
            # 'GetSubstructMatch',
            # 'GetSubstructMatches',
            'GetUnsignedProp',
            'HasProp',
            'HasSubstructMatch',
            'InsertMol',
            'NeedsUpdatePropertyCache',
            'RemoveAllConformers',
            'RemoveAtom',
            'RemoveBond',
            'RemoveConformer',
            'ReplaceAtom',
            'ReplaceBond',
            'RollbackBatchEdit',
            'SetBoolProp',
            'SetDoubleProp',
            'SetIntProp',
            'SetProp',
            'SetStereoGroups',
            'SetUnsignedProp',
            'ToBinary',
            'UpdatePropertyCache',]

        # Update the list of methods based on the actual RDKit Mol object
        for idx in range(len(list_of_methods) - 1, -1, -1):
            if not hasattr(rdkit_mol, list_of_methods[idx]):
                list_of_methods.pop(idx)

        # Check if methods are inherited
        for method in list_of_methods:
            # If RDKitMol has this method
            self.assertTrue(hasattr(mol, method))
            # Check if these methods are directly link to the original method in the backend molecule
            self.assertEqual(getattr(mol, method),
                             getattr(rdkit_mol, method))

    def test_smiles_without_atom_mapping_and_hs(self):
        """
        Test exporting a molecule as a SMILES string without atom mapping and explicit H atoms.
        """
        test_strings = ['[C-]#[O+]', '[C]', '[CH]', 'OO', '[H][H]', '[H]',
                        '[He]', '[O]', 'O', '[CH3]', 'C', '[OH]', 'CCC',
                        'CC', 'N#N', '[O]O', '[CH2]C', '[Ar]', 'CCCC',
                        'O=C=O', '[C]#N',
                        ]
        for s in test_strings:
            molecule = RDKitMol.FromSmiles(s)
            self.assertEqual(s, molecule.ToSmiles())

    def test_smiles_with_atom_mapping_and_hs(self):
        """
        Test exporting a molecule as a SMILES string with atom mapping and explicit H atoms.
        """
        # Normal SMILES without atom mapping, atommap and H atoms will be
        # assigned during initiation
        mol1 = RDKitMol.FromSmiles('[CH2]C')
        # Export SMILES with H atoms
        self.assertEqual(mol1.ToSmiles(removeHs=False,),
                         '[H][C]([H])C([H])([H])[H]')
        # Export SMILES with H atoms and indexes
        self.assertEqual(mol1.ToSmiles(removeHs=False, removeAtomMap=False),
                         '[C:1]([C:2]([H:5])([H:6])[H:7])([H:3])[H:4]')

        # SMILES with atom mapping
        mol2 = RDKitMol.FromSmiles('[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]')
        # Test the atom indexes and atom map numbers share the same order
        self.assertSequenceEqual(mol2.GetAtomMapNumbers(),
                                 (1, 2, 3, 4, 5, 6, 7))
        # Test the 2nd and 4th atoms are carbons
        self.assertEqual(mol2.GetAtomWithIdx(1).GetAtomicNum(), 6)
        self.assertEqual(mol2.GetAtomWithIdx(3).GetAtomicNum(), 6)
        # Export SMILES without H atoms and atom map
        self.assertEqual(mol2.ToSmiles(), '[CH2]C')
        # Export SMILES with H atoms and without atom map
        self.assertEqual(mol2.ToSmiles(removeHs=False,),
                         '[H][C]([H])C([H])([H])[H]')
        # Export SMILES without H atoms and with atom map
        # Atom map numbers for heavy atoms are perserved
        self.assertEqual(mol2.ToSmiles(removeAtomMap=False,),
                         '[CH3:2][CH2:4]')
        # Export SMILES with H atoms and with atom map
        self.assertEqual(mol2.ToSmiles(removeHs=False, removeAtomMap=False,),
                         '[H:1][C:4]([C:2]([H:5])([H:6])[H:7])[H:3]')

        # SMILES with atom mapping but neglect the atom mapping
        mol3 = RDKitMol.FromSmiles('[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]', keepAtomMap=False)
        # Test the atom indexes and atom map numbers share the same order
        self.assertSequenceEqual(mol3.GetAtomMapNumbers(),
                                 (1, 2, 3, 4, 5, 6, 7))
        # However, now the 4th atom is not carbon (3rd instead), and atom map numbers
        # are determined by the sequence of atom appear in the SMILES.
        self.assertEqual(mol3.GetAtomWithIdx(1).GetAtomicNum(), 6)
        self.assertEqual(mol3.GetAtomWithIdx(2).GetAtomicNum(), 6)
        # Export SMILES with H atoms and with atom map
        self.assertEqual(mol3.ToSmiles(removeHs=False, removeAtomMap=False,),
                         '[H:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]')

        # SMILES with uncommon atom mapping starting from 0 and being discontinue
        mol4 = RDKitMol.FromSmiles('[H:9][C:2]([C:5]([H:0])[H:3])([H:4])[H:8]')
        # Test the atom indexes and atom map numbers share the same order
        self.assertSequenceEqual(mol4.GetAtomMapNumbers(),
                                 (0, 2, 3, 4, 5, 8, 9))
        # Check Heavy atoms' index
        self.assertEqual(mol4.GetAtomWithIdx(1).GetAtomicNum(), 6)
        self.assertEqual(mol4.GetAtomWithIdx(4).GetAtomicNum(), 6)
                # Export SMILES without H atoms and with atom map
        # Atom map numbers for heavy atoms are perserved
        self.assertEqual(mol4.ToSmiles(removeAtomMap=False,),
                         '[CH3:2][CH2:5]')
        # Export SMILES with H atoms and with atom map
        self.assertEqual(mol4.ToSmiles(removeHs=False, removeAtomMap=False,),
                         '[H:0][C:5]([C:2]([H:4])([H:8])[H:9])[H:3]')

    def test_generate_mol_from_openbabel_mol(self):
        """
        Test generating the RDKitMol object from an Openbabel Molecule object.
        """
        # Generate from openbabel without embedded geometries
        pmol = pybel.readstring('smi', self.smi1)
        pmol.addh()
        ob_mol = pmol.OBMol
        mol1 = RDKitMol.FromOBMol(ob_mol)  # default arguments
        self.assertEqual(mol1.ToSmiles(), 'CC')
        self.assertEqual(mol1.GetNumConformers(), 0)

        # Generate from OBMol with geometries
        pmol = pybel.readstring('smi', 'CCC')
        pmol.addh()
        pmol.make3D()
        ob_mol = pmol.OBMol
        mol2 = RDKitMol.FromOBMol(ob_mol)
        self.assertEqual(mol2.ToSmiles(), 'CCC')
        self.assertEqual(mol2.GetNumConformers(), 1)
        self.assertEqual(mol2.GetPositions().shape, (11, 3))

    def test_generate_from_rmg_mol(self):
        """
        Test generate from rmg molecule(self):
        """
        # Currently, it requires install RMG as a backend
        # But it is not a default package
        # So currently not included in the unittest
        pass

    def test_add_redundant_bonds(self):
        """
        Test adding redundant bond to a molecule.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'

        # Add a redundant bond (C2-H3) that exists in the molecule
        # This should raise an error
        mol1 = RDKitMol.FromSmiles(smi)
        self.assertIsNotNone(mol1.GetBondBetweenAtoms(1,2))
        self.assertRaises(RuntimeError, mol1.AddRedundantBonds, [(1,2)])

        # Add a bond between (H1-H3)
        mol2 = RDKitMol.FromSmiles(smi)
        mol_w_new_bond = mol2.AddRedundantBonds([(0,2)])
        # mol_w_new_bond should be different mol objects
        self.assertNotEqual(mol_w_new_bond, mol2)
        self.assertNotEqual(mol_w_new_bond._mol, mol2._mol)
        # The new mol should contain the new bond
        self.assertIsNone(mol2.GetBondBetweenAtoms(0,2))
        new_bond = mol_w_new_bond.GetBondBetweenAtoms(0,2)
        self.assertIsNotNone(new_bond)
        # The redundant bond has a bond order of 1.0
        self.assertEqual(new_bond.GetBondTypeAsDouble(), 1.0)

    def test_get_atomic_numbers(self):
        """
        Test getAtomicNumbers returns a list of atomic numbers corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertSequenceEqual(mol.GetAtomicNumbers(),
                                 [1,6,1,1,1,1,])

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertSequenceEqual(mol.GetAtomicNumbers(),
                                 [8,6,6,1,1,1,1])

    def test_get_element_symbols(self):
        """
        Test getAtomicNumbers returns a list of elementary symbols corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertSequenceEqual(mol.GetElementSymbols(),
                                 ['H','C','H','H','H','H',])

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertSequenceEqual(mol.GetElementSymbols(),
                                 ['O','C','C','H','H','H','H'])

    def test_get_atom_masses(self):
        """
        Test getAtomMasses returns a list of atom mass corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertTrue(np.all(np.isclose(
                                mol.GetAtomMasses(),
                                [1.008, 12.011, 1.008, 1.008, 1.008, 1.008],
                                atol=1e-2, rtol=1e-3)))

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertTrue(np.all(np.isclose(
                                mol.GetAtomMasses(),
                                [15.999, 12.011, 12.011, 1.008, 1.008, 1.008, 1.008],
                                atol=1e-2, rtol=1e-3)))

    def test_get_bond_as_tuples(self):
        """
        Test getBondsAsTuples returns a list of atom pairs corresponding to each bond.
        """
        # Single molecule
        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertCountEqual(mol.GetBondsAsTuples(),
                              [(0, 1), (1, 2), (1, 5), (1, 6), (2, 3), (2, 4)])

        # Mol fragments
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        self.assertCountEqual(mol.GetBondsAsTuples(),
                              [(1, 2), (1, 3), (1, 4), (1, 5)])

    def test_get_torsion_tops(self):
        """
        Test get torsion tops of a given molecule.
        """
        smi1 = '[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]'
        mol = RDKitMol.FromSmiles(smi1)
        tops = mol.GetTorsionTops([2, 0, 1, 5])
        self.assertCountEqual(tops, ((0, 2, 3, 4), (1, 5, 6, 7)))

        smi2 = '[C:1]([C:2]#[C:3][C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi2)
        with self.assertRaises(ValueError):
            mol.GetTorsionTops([4, 0, 3, 7])
        tops = mol.GetTorsionTops([4, 0, 3, 7], allowNonbondPivots=True)
        self.assertCountEqual(tops, ((0, 4, 5, 6), (3, 7, 8, 9)))

        smi3 = '[C:1]([H:3])([H:4])([H:5])[H:6].[O:2][H:7]'
        mol = RDKitMol.FromSmiles(smi3)
        mol = mol.AddRedundantBonds([[1, 2]])
        with self.assertRaises(ValueError):
            mol.GetTorsionTops([3, 0, 1, 6])
        tops = mol.GetTorsionTops([3, 0, 1, 6], allowNonbondPivots=True)
        self.assertCountEqual(tops, ((0, 3, 4, 5), (1, 6)))

    def test_combined_mol(self):
        """
        Test combining molecules using CombineMol.
        """
        xyz_1 = np.array([[-0.01841209, -0.00118705,  0.00757447],
                          [-0.66894707, -0.81279485, -0.34820667],
                          [-0.36500814,  1.00785186, -0.31659064],
                          [ 0.08216461, -0.04465528,  1.09970299],
                          [ 0.97020269, -0.14921467, -0.44248015]])
        xyz_2 = np.array([[ 0.49911347,  0.        ,  0.        ],
                          [-0.49911347,  0.        ,  0.        ]])
        m1 = RDKitMol.FromSmiles('C')
        m1.EmbedConformer()
        m1.SetPositions(xyz_1)

        m2 = RDKitMol.FromSmiles('[OH]')
        m2.EmbedConformer()
        m2.SetPositions(xyz_2)

        combined = m1.CombineMol(m2)
        self.assertTrue(np.allclose(np.concatenate([xyz_1, xyz_2],),
                                    combined.GetPositions()))

        combined = m1.CombineMol(m2, 1.)
        self.assertTrue(np.allclose(np.concatenate([xyz_1, xyz_2+np.array([[1., 0., 0.]])],),
                                    combined.GetPositions()))

        combined = m1.CombineMol(m2, np.array([[1., 1., 0.]]))
        self.assertTrue(np.allclose(np.concatenate([xyz_1, xyz_2+np.array([[1., 1., 0.]])],),
                                    combined.GetPositions()))

        combined = m2.CombineMol(m1, np.array([[0., 0., 1.]]))
        self.assertTrue(np.allclose(np.concatenate([xyz_2, xyz_1+np.array([[0., 0., 1.]])],),
                                    combined.GetPositions()))

        m1.EmbedMultipleConfs(10)
        m2.RemoveAllConformers()
        self.assertEqual(10, m1.CombineMol(m2).GetNumConformers())
        self.assertEqual(10, m2.CombineMol(m1).GetNumConformers())
        self.assertEqual(0, m1.CombineMol(m2, c_product=True).GetNumConformers())
        self.assertEqual(0, m2.CombineMol(m1, c_product=True).GetNumConformers())

        m2.EmbedMultipleConfs(10)
        self.assertEqual(10, m1.CombineMol(m2).GetNumConformers())
        self.assertEqual(10, m2.CombineMol(m1).GetNumConformers())
        self.assertEqual(100, m1.CombineMol(m2, c_product=True).GetNumConformers())
        self.assertEqual(100, m2.CombineMol(m1, c_product=True).GetNumConformers())

        m2.EmbedMultipleConfs(20)
        self.assertEqual(10, m1.CombineMol(m2).GetNumConformers())
        self.assertEqual(20, m2.CombineMol(m1).GetNumConformers())
        self.assertEqual(200, m1.CombineMol(m2, c_product=True).GetNumConformers())
        self.assertEqual(200, m2.CombineMol(m1, c_product=True).GetNumConformers())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
