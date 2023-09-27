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

from rdmc import (get_unique_mols,
                  has_matched_mol,
                  parse_xyz_or_smiles_list,
                  RDKitMol)
import pytest

logging.basicConfig(level=logging.DEBUG)

################################################################################


class TestRDKitMol:
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
        assert mol._mol == rdkit_mol
        assert mol.ToRWMol() == rdkit_mol

        # Option 2
        mol = RDKitMol.FromMol(rdkit_mol, keepAtomMap=False)
        # The backend molecule should be the same as the input RWMol object
        assert mol._mol == rdkit_mol
        assert mol.ToRWMol() == rdkit_mol

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
            # 'GetAtoms',
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
            assert hasattr(mol, method)
            # Check if these methods are directly link to the original method in the backend molecule
            assert getattr(mol, method) == getattr(rdkit_mol, method)

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
            assert s == molecule.ToSmiles()

    def test_smiles_with_atom_mapping_and_hs(self):
        """
        Test exporting a molecule as a SMILES string with atom mapping and explicit H atoms.
        """
        # Normal SMILES without atom mapping, atommap and H atoms will be
        # assigned during initiation
        mol1 = RDKitMol.FromSmiles('[CH2]C')
        # Export SMILES with H atoms
        assert mol1.ToSmiles(removeHs=False,) == '[H][C]([H])C([H])([H])[H]'
        # Export SMILES with H atoms and indexes
        assert mol1.ToSmiles(removeHs=False,
                             removeAtomMap=False) \
            == '[C:1]([C:2]([H:5])([H:6])[H:7])([H:3])[H:4]'

        # SMILES with atom mapping
        mol2 = RDKitMol.FromSmiles('[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]')
        # Test the atom indexes and atom map numbers share the same order
        assert mol2.GetAtomMapNumbers() == (1, 2, 3, 4, 5, 6, 7)
        # Test the 2nd and 4th atoms are carbons
        assert mol2.GetAtomWithIdx(1).GetAtomicNum() == 6
        assert mol2.GetAtomWithIdx(3).GetAtomicNum() == 6
        # Export SMILES without H atoms and atom map
        assert mol2.ToSmiles() == '[CH2]C'
        # Export SMILES with H atoms and without atom map
        assert mol2.ToSmiles(removeHs=False,) == '[H][C]([H])C([H])([H])[H]'
        # Export SMILES without H atoms and with atom map
        # Atom map numbers for heavy atoms are perserved
        assert mol2.ToSmiles(removeAtomMap=False,) == '[CH3:2][CH2:4]'
        # Export SMILES with H atoms and with atom map
        assert mol2.ToSmiles(removeHs=False, removeAtomMap=False,) == '[H:1][C:4]([C:2]([H:5])([H:6])[H:7])[H:3]'

        # SMILES with atom mapping but neglect the atom mapping
        mol3 = RDKitMol.FromSmiles('[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]', keepAtomMap=False)
        # Test the atom indexes and atom map numbers share the same order
        assert mol3.GetAtomMapNumbers() == (1, 2, 3, 4, 5, 6, 7)
        # However, now the 4th atom is not carbon (3rd instead), and atom map numbers
        # are determined by the sequence of atom appear in the SMILES.
        assert mol3.GetAtomWithIdx(1).GetAtomicNum() == 6
        assert mol3.GetAtomWithIdx(2).GetAtomicNum() == 6
        # Export SMILES with H atoms and with atom map
        assert mol3.ToSmiles(removeHs=False,
                             removeAtomMap=False,) \
            == '[H:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'

        # SMILES with uncommon atom mapping starting from 0 and being discontinue
        mol4 = RDKitMol.FromSmiles('[H:9][C:2]([C:5]([H:0])[H:3])([H:4])[H:8]')
        # Test the atom indexes and atom map numbers share the same order
        assert mol4.GetAtomMapNumbers() == (0, 2, 3, 4, 5, 8, 9)
        # Check Heavy atoms' index
        assert mol4.GetAtomWithIdx(1).GetAtomicNum() == 6
        assert mol4.GetAtomWithIdx(4).GetAtomicNum() == 6
        # Export SMILES without H atoms and with atom map
        # Atom map numbers for heavy atoms are perserved
        assert mol4.ToSmiles(removeAtomMap=False,) == '[CH3:2][CH2:5]'
        # Export SMILES with H atoms and with atom map
        assert mol4.ToSmiles(removeHs=False, removeAtomMap=False) == '[H:0][C:5]([C:2]([H:4])([H:8])[H:9])[H:3]'

    def test_generate_mol_from_openbabel_mol(self):
        """
        Test generating the RDKitMol object from an Openbabel Molecule object.
        """
        # Generate from openbabel without embedded geometries
        pmol = pybel.readstring('smi', self.smi1)
        pmol.addh()
        ob_mol = pmol.OBMol
        mol1 = RDKitMol.FromOBMol(ob_mol)  # default arguments
        assert mol1.ToSmiles() == 'CC'
        assert mol1.GetNumConformers() == 0

        # Generate from OBMol with geometries
        pmol = pybel.readstring('smi', 'CCC')
        pmol.addh()
        pmol.make3D()
        ob_mol = pmol.OBMol
        mol2 = RDKitMol.FromOBMol(ob_mol)
        assert mol2.ToSmiles() == 'CCC'
        assert mol2.GetNumConformers() == 1
        assert mol2.GetPositions().shape == (11, 3)

    def test_generate_mol_from_rmg_mol(self):
        """
        Test generate mol from rmg molecule(self):
        """
        # Currently, it requires install RMG as a backend
        # But it is not a default package
        # So currently not included in the unittest
        pass

    def test_generate_mol_from_inchi(self):
        """
        Test generate the RDKitMol from InChI strings.
        """
        # InChI of a stable species
        inchi1 = 'InChI=1S/H2O/h1H2'
        mol1 = RDKitMol.FromInchi(inchi1)
        assert mol1.GetNumAtoms() == 3
        assert mol1.GetAtomicNumbers() == [8, 1, 1]
        assert set(mol1.GetBondsAsTuples()) == {(0, 1), (0, 2)}

        # The case of addHs == False
        mol2 = RDKitMol.FromInchi(inchi1, addHs=False)
        assert mol2.GetNumAtoms() == 1
        assert mol2.GetAtomicNumbers() == [8]
        assert mol2.GetAtomWithIdx(0).GetNumExplicitHs() == 2

        # InChI of a radical
        inchi2 = 'InChI=1S/CH3/h1H3'
        mol3 = RDKitMol.FromInchi(inchi2)
        assert mol3.GetNumAtoms() == 4
        assert mol3.GetAtomicNumbers() == [6, 1, 1, 1]
        assert mol3.GetAtomWithIdx(0).GetNumRadicalElectrons() == 1
        assert set(mol3.GetBondsAsTuples()) == {(0, 1), (0, 2), (0, 3)}

        # InChI of an anion
        inchi3 = 'InChI=1S/H2O/h1H2/p-1'
        mol4 = RDKitMol.FromInchi(inchi3)
        assert mol4.GetNumAtoms() == 2
        assert mol4.GetAtomicNumbers() == [8, 1]
        assert mol4.GetAtomWithIdx(0).GetFormalCharge() == -1
        assert set(mol4.GetBondsAsTuples()) == {(0, 1)}

        # InChI of an cation
        inchi4 = 'InChI=1S/H3N/h1H3/p+1'
        mol5 = RDKitMol.FromInchi(inchi4)
        assert mol5.GetNumAtoms() == 5
        assert mol5.GetAtomicNumbers() == [7, 1, 1, 1, 1]
        assert mol5.GetAtomWithIdx(0).GetFormalCharge() == 1
        assert set(mol5.GetBondsAsTuples()) == {(0, 1), (0, 2), (0, 3), (0, 4)}

        # InChI of benzene (aromatic ring)
        inchi5 = 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H'
        mol6 = RDKitMol.FromInchi(inchi5)
        assert len(mol6.GetAromaticAtoms()) == 6

        # Todo: check stereochemistry

    def test_mol_to_inchi(self):
        """
        Test converting RDKitMol to InChI strings.
        """
        for inchi in ['InChI=1S/H2O/h1H2',
                      'InChI=1S/CH3/h1H3',
                      'InChI=1S/H2O/h1H2/p-1',
                      'InChI=1S/H3N/h1H3/p+1',
                      'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H',
                      ]:
            assert inchi == RDKitMol.FromInchi(inchi).ToInchi()

    def test_mol_to_xyz(self):
        """
        Test converting RDKitMol to XYZ strings.
        """
        xyz = """1\n\nH      0.000000    0.000000    0.000000\n"""
        mol = RDKitMol.FromXYZ(xyz)
        assert mol.ToXYZ(header=True) == xyz
        assert mol.ToXYZ(header=False) == xyz[3:]  # Currently to XYZ without header has no line break at the end
        assert mol.ToXYZ(header=True, comment='test') == """1\ntest\nH      0.000000    0.000000    0.000000\n"""

    def test_add_redundant_bonds(self):
        """
        Test adding redundant bond to a molecule.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'

        # Add a redundant bond (C2-H3) that exists in the molecule
        # This should raise an error
        mol1 = RDKitMol.FromSmiles(smi)
        assert mol1.GetBondBetweenAtoms(1, 2) is not None
        with pytest.raises(RuntimeError):
            mol1.AddRedundantBonds([(1, 2)])

        # Add a bond between (H1-H3)
        mol2 = RDKitMol.FromSmiles(smi)
        mol_w_new_bond = mol2.AddRedundantBonds([(0, 2)])
        # mol_w_new_bond should be different mol objects
        assert mol_w_new_bond != mol2
        assert mol_w_new_bond._mol != mol2._mol
        # The new mol should contain the new bond
        assert mol2.GetBondBetweenAtoms(0, 2) is None
        new_bond = mol_w_new_bond.GetBondBetweenAtoms(0, 2)
        assert new_bond is not None
        # The redundant bond has a bond order of 1.0
        assert new_bond.GetBondTypeAsDouble() == 1.0

    def test_add_null_conformer(self):
        """
        Test adding null conformers to a molecule.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        assert mol.GetNumConformers() == 0

        # Test adding 1 null conformer with default arguments
        mol.AddNullConformer()
        assert mol.GetNumConformers() == 1
        conf = mol.GetConformer(id=0)
        coords = conf.GetPositions()
        assert coords.shape == (6, 3)
        # Test all atom coordinates are non-zero
        assert np.all(np.all(coords != np.zeros((6, 3)), axis=1))

        # Test adding a conformer with a specific id and all zero
        mol.AddNullConformer(confId=10, random=False)
        assert mol.GetNumConformers() == 2
        with pytest.raises(ValueError):
            mol.GetConformer(id=1)
        conf = mol.GetConformer(id=10)
        coords = conf.GetPositions()
        assert coords.shape == (6, 3)
        assert np.all(np.equal(coords, np.zeros((6, 3))))

    def test_get_atoms(self):
        """
        Test the rewrite version of GetAtoms returns the same results as Mol.GetAtoms.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])
        assert np.all([atom1.GetIdx() == atom2.GetIdx()
                       for atom1, atom2 in zip(mol.GetAtoms(), mol._mol.GetAtoms())])

    def test_get_atomic_numbers(self):
        """
        Test getAtomicNumbers returns a list of atomic numbers corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        assert mol.GetAtomicNumbers() == [1, 6, 1, 1, 1, 1,]

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        assert mol.GetAtomicNumbers() == [8, 6, 6, 1, 1, 1, 1]

    def test_get_element_symbols(self):
        """
        Test getAtomicNumbers returns a list of elementary symbols corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        assert mol.GetElementSymbols() == ['H', 'C', 'H', 'H', 'H', 'H',]

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        assert mol.GetElementSymbols() == ['O', 'C', 'C', 'H', 'H', 'H', 'H']

    def test_get_atom_masses(self):
        """
        Test getAtomMasses returns a list of atom mass corresponding to each atom in the atom index order.
        """
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        assert np.all(
            np.isclose(
                mol.GetAtomMasses(),
                [1.008, 12.011, 1.008, 1.008, 1.008, 1.008],
                atol=1e-2,
                rtol=1e-3)
        )

        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        assert np.all(
            np.isclose(
                mol.GetAtomMasses(),
                [15.999, 12.011, 12.011, 1.008, 1.008, 1.008, 1.008],
                atol=1e-2,
                rtol=1e-3)
        )

    def test_get_bond_as_tuples(self):
        """
        Test getBondsAsTuples returns a list of atom pairs corresponding to each bond.
        """
        # Single molecule
        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        bonds = mol.GetBondsAsTuples()
        assert len(bonds) == 6
        assert set(bonds) == {(0, 1), (1, 2), (1, 5), (1, 6), (2, 3), (2, 4)}

        # Mol fragments
        smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
        mol = RDKitMol.FromSmiles(smi)
        bonds = mol.GetBondsAsTuples()
        assert len(bonds) == 4
        assert set(bonds) == {(1, 2), (1, 3), (1, 4), (1, 5)}

    def test_get_torsion_tops(self):
        """
        Test get torsion tops of a given molecule.
        """
        smi1 = '[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]'
        mol = RDKitMol.FromSmiles(smi1)
        tops = mol.GetTorsionTops([2, 0, 1, 5])
        assert len(tops) == 2
        assert set(tops) == {(0, 2, 3, 4), (1, 5, 6, 7)}

        smi2 = '[C:1]([C:2]#[C:3][C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi2)
        with pytest.raises(ValueError):
            mol.GetTorsionTops([4, 0, 3, 7])
        tops = mol.GetTorsionTops([4, 0, 3, 7], allowNonbondPivots=True)
        assert len(tops) == 2
        assert set(tops) == {(0, 4, 5, 6), (3, 7, 8, 9)}

        smi3 = '[C:1]([H:3])([H:4])([H:5])[H:6].[O:2][H:7]'
        mol = RDKitMol.FromSmiles(smi3)
        mol = mol.AddRedundantBonds([[1, 2]])
        with pytest.raises(ValueError):
            mol.GetTorsionTops([3, 0, 1, 6])
        tops = mol.GetTorsionTops([3, 0, 1, 6], allowNonbondPivots=True)
        assert len(tops) == 2
        assert set(tops) == {(0, 3, 4, 5), (1, 6)}

    def test_combined_mol(self):
        """
        Test combining molecules using CombineMol.
        """
        xyz_1 = np.array([[-0.01841209, -0.00118705, 0.00757447],
                          [-0.66894707, -0.81279485, -0.34820667],
                          [-0.36500814, 1.00785186, -0.31659064],
                          [0.08216461, -0.04465528, 1.09970299],
                          [0.97020269, -0.14921467, -0.44248015]])
        xyz_2 = np.array([[0.49911347, 0., 0.],
                          [-0.49911347, 0., 0.]])
        m1 = RDKitMol.FromSmiles('C')
        m1.EmbedConformer()
        m1.SetPositions(xyz_1)

        m2 = RDKitMol.FromSmiles('[OH]')
        m2.EmbedConformer()
        m2.SetPositions(xyz_2)

        combined = m1.CombineMol(m2)
        assert np.allclose(np.concatenate([xyz_1, xyz_2],),
                           combined.GetPositions())

        combined = m1.CombineMol(m2, 1.)
        assert np.allclose(np.concatenate([xyz_1,
                                           xyz_2 + np.array([[1., 0., 0.]])],),
                           combined.GetPositions())

        combined = m1.CombineMol(m2, np.array([[1., 1., 0.]]))
        assert np.allclose(np.concatenate([xyz_1,
                                           xyz_2 + np.array([[1., 1., 0.]])],),
                           combined.GetPositions())

        combined = m2.CombineMol(m1, np.array([[0., 0., 1.]]))
        assert np.allclose(np.concatenate([xyz_2,
                                           xyz_1 + np.array([[0., 0., 1.]])],),
                           combined.GetPositions())

        m1.EmbedMultipleConfs(10)
        m2.RemoveAllConformers()
        assert 10 == m1.CombineMol(m2).GetNumConformers()
        assert 10 == m2.CombineMol(m1).GetNumConformers()
        assert 0 == m1.CombineMol(m2, c_product=True).GetNumConformers()
        assert 0 == m2.CombineMol(m1, c_product=True).GetNumConformers()

        m2.EmbedMultipleConfs(10)
        assert 10 == m1.CombineMol(m2).GetNumConformers()
        assert 10 == m2.CombineMol(m1).GetNumConformers()
        assert 100 == m1.CombineMol(m2, c_product=True).GetNumConformers()
        assert 100 == m2.CombineMol(m1, c_product=True).GetNumConformers()

        m2.EmbedMultipleConfs(20)
        assert 10 == m1.CombineMol(m2).GetNumConformers()
        assert 20 == m2.CombineMol(m1).GetNumConformers()
        assert 200 == m1.CombineMol(m2, c_product=True).GetNumConformers()
        assert 200 == m2.CombineMol(m1, c_product=True).GetNumConformers()

    def test_renumber_atoms(self):
        """
        Test the functionality of renumber atoms of a molecule.
        """
        # A molecule complex
        smi = '[C:1]([C:2]([C:3]([H:20])([H:21])[H:22])([O:4])[C:5]([H:23])([H:24])[H:25])' \
              '([H:17])([H:18])[H:19].[C:6]([C:7]([C:8]([H:29])([H:30])[H:31])([C:9]([H:32])' \
              '([H:33])[H:34])[c:10]1[c:11]([H:35])[c:12]([H:36])[c:13]([O:14][H:37])[c:15]' \
              '([H:38])[c:16]1[H:39])([H:26])([H:27])[H:28]'

        # The generated molecule will maintain all the atom map numbers and the atom indexes
        # have the same sequence as the atom map numbers
        ref_mol = RDKitMol.FromSmiles(smi, keepAtomMap=True)

        # Since the molecule atom indexes are consistent with the atom map numbers
        # The generated molecule should have the same atom map numbers
        assert ref_mol.RenumberAtoms(updateAtomMap=False).GetAtomMapNumbers() \
            == ref_mol.GetAtomMapNumbers()

        # Create a molecule with different atom indexes
        mols = [RDKitMol.FromSmiles(smi, keepAtomMap=True) for smi in smi.split('.')]
        combined = mols[0].CombineMol(mols[1])
        # If not renumbered, then atom maps and atom sequences are different
        assert combined.GetAtomMapNumbers() != ref_mol.GetAtomMapNumbers()
        assert combined.GetAtomicNumbers() != ref_mol.GetAtomicNumbers()
        # Atom maps and atom sequences are the same to the reference molecule now
        assert combined.RenumberAtoms(updateAtomMap=False).GetAtomMapNumbers() \
            == ref_mol.GetAtomMapNumbers()
        assert combined.RenumberAtoms(updateAtomMap=False).GetAtomicNumbers() \
            == ref_mol.GetAtomicNumbers()

        smi = '[C:1]([H:2])([H:3])([H:4])[H:5]'
        ref_mol = RDKitMol.FromSmiles(smi)
        # Renumber molecule but keep the original atom map
        renumbered = ref_mol.RenumberAtoms([1, 2, 3, 4, 0], updateAtomMap=False)
        assert renumbered.GetAtomMapNumbers() == (2, 3, 4, 5, 1)
        assert renumbered.GetAtomicNumbers() == [1, 1, 1, 1, 6]
        # Renumber molecule but also update the atom map after renumbering
        renumbered = ref_mol.RenumberAtoms([1, 2, 3, 4, 0], updateAtomMap=True)
        assert renumbered.GetAtomMapNumbers() == (1, 2, 3, 4, 5)
        assert renumbered.GetAtomicNumbers() == [1, 1, 1, 1, 6]

    def test_copy(self):
        """
        Test copy molecule functionality.
        """
        smi = '[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]'
        mol = RDKitMol.FromSmiles(smi)
        mol.EmbedConformer()

        mol_copy = mol.Copy()
        assert mol.__hash__() != mol_copy.__hash__()
        assert mol.GetAtomicNumbers() == mol_copy.GetAtomicNumbers()
        assert mol.GetNumConformers() == mol_copy.GetNumConformers()
        assert np.allclose(mol.GetPositions(),
                           mol_copy.GetPositions())

        mol_copy = mol.Copy(quickCopy=True)
        assert mol.__hash__() != mol_copy.__hash__()
        assert mol.GetAtomicNumbers() == mol_copy.GetAtomicNumbers()
        assert mol_copy.GetNumConformers() == 0

    def test_has_matched_mol(self):
        """
        Test the function that indicates if there is a matched molecule to the query molecule from a list.
        """
        query = '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'

        list1 = ['C', 'O']
        list2 = ['C', 'O', '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]']
        list3 = ['C', 'O', '[C:1]([O:6][H:2])([H:3])([H:4])[H:5]']

        assert not has_matched_mol(
            RDKitMol.FromSmiles(query),
            [RDKitMol.FromSmiles(smi) for smi in list1],
        )
        assert has_matched_mol(
            RDKitMol.FromSmiles(query),
            [RDKitMol.FromSmiles(smi) for smi in list2],
        )
        assert has_matched_mol(
            RDKitMol.FromSmiles(query),
            [RDKitMol.FromSmiles(smi) for smi in list2],
            consider_atommap=True
        )
        assert has_matched_mol(
            RDKitMol.FromSmiles(query),
            [RDKitMol.FromSmiles(smi) for smi in list3],
        )
        assert not has_matched_mol(
            RDKitMol.FromSmiles(query),
            [RDKitMol.FromSmiles(smi) for smi in list3],
            consider_atommap=True
        )

    def test_get_unique_mols(self):
        """
        Test the function that extract unique molecules from a list of molecules.
        """
        list1 = ['C', 'O']
        list2 = ['C', 'O',
                 '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]',
                 '[C:1]([H:3])([H:4])([H:5])[O:6][H:2]',]

        assert len(get_unique_mols(
            [RDKitMol.FromSmiles(smi) for smi in list1]
        )) == 2
        assert set([mol.ToSmiles() for mol in get_unique_mols(
            [RDKitMol.FromSmiles(smi) for smi in list1]
        )]) == {'C', 'O'}
        assert len(get_unique_mols(
            [RDKitMol.FromSmiles(smi) for smi in list2],
            consider_atommap=True
        )) == 4
        assert set(
            [mol.ToSmiles(removeHs=False, removeAtomMap=False)
             for mol in get_unique_mols(
                [RDKitMol.FromSmiles(smi) for smi in list2],
                consider_atommap=True)]) \
            == {
            '[O:1]([H:2])[H:3]',
            '[C:1]([H:2])([H:3])([H:4])[H:5]',
            '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]',
            '[C:1]([H:3])([H:4])([H:5])[O:6][H:2]',
        }
        assert len(get_unique_mols(
            [RDKitMol.FromSmiles(smi) for smi in list2],
            consider_atommap=False
        )) == 3

    def test_saturate_biradical_site12(self):
        """
        Test the function that saturates the adjacent biradical sites.
        """
        smi = '[CH2][CH2]'
        mol = RDKitMol.FromSmiles(smi)

        assert mol.GetSpinMultiplicity() == 3
        # Test the case where no action is needed
        mol.SaturateBiradicalSites12(multiplicity=3, verbose=True)
        assert mol.GetSpinMultiplicity() == 3
        mol.SaturateBiradicalSites12(multiplicity=4, verbose=True)
        assert mol.GetSpinMultiplicity() == 3
        # Test the case where molecule cannot be saturated to the given multiplicity
        mol.SaturateBiradicalSites12(multiplicity=2, verbose=True)
        assert mol.GetSpinMultiplicity() == 3

        # Test the case where molecule can be saturated to the given multiplicity
        mol.SaturateBiradicalSites12(multiplicity=1, verbose=True)
        assert mol.GetSpinMultiplicity() == 1

    def test_saturate_biradical_sites_conjugated_double_bond(self):
        """
        Test the function that saturates the biradicals that have conjugated double bond.
        """
        smi = '[CH2]C=C[CH2]'
        mol = RDKitMol.FromSmiles(smi)

        assert mol.GetSpinMultiplicity() == 3
        # Test the case where no action is needed
        mol.SaturateBiradicalSitesCDB(multiplicity=3, verbose=True)
        assert mol.GetSpinMultiplicity() == 3
        mol.SaturateBiradicalSitesCDB(multiplicity=4, verbose=True)
        assert mol.GetSpinMultiplicity() == 3
        # Test the case where molecule cannot be saturated to the given multiplicity
        mol.SaturateBiradicalSitesCDB(multiplicity=2, verbose=True)
        assert mol.GetSpinMultiplicity() == 3

        # Test the case where molecule can be saturated to the given multiplicity
        mol.SaturateBiradicalSitesCDB(multiplicity=1, verbose=True)
        assert mol.GetSpinMultiplicity() == 1

    def test_get_finger_print(self):
        """
        Test the function that generates molecular finger prints.
        """
        # We only test one case here to check the functionality of the function
        # other cases are covered by test_fingerprints
        smi = 'O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1'
        fp = RDKitMol.FromSmiles(smi, addHs=False).GetFingerprint(fpType='morgan', numBits=2048, count=True, radius=3)
        fp_expect \
            = Chem.rdFingerprintGenerator \
            .GetMorganGenerator(radius=3, fpSize=2048)\
            .GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smi))
        assert np.isclose(fp, fp_expect).all()


def test_parse_xyz_or_smiles_list():
    """
    Test the function that parses a list of xyz or smiles strings.
    """
    mols = parse_xyz_or_smiles_list(
        ['CCC', 'H 0 0 0', ('[CH2]', 1)],
        header=False,
        backend='jensen',
    )
    assert len(mols) == 3
    assert mols[0].ToSmiles() == 'CCC'
    assert mols[1].ToSmiles() == '[H]'
    assert mols[2].ToSmiles() == '[CH2]'
    assert mols[2].GetSpinMultiplicity() == 1

    mols, is_3D = parse_xyz_or_smiles_list(
        ['CCC', 'H 0 0 0', ('[CH2]', 1)],
        header=False,
        backend='jensen',
        with_3d_info=True)
    assert is_3D == [False, True, False]


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
