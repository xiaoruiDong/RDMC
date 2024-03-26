import copy

import pytest
import numpy as np
from rdmc import RDKitMol


@pytest.mark.parametrize(
    'rad_smi, expect_smi',
    [
        ('[CH3]', 'C'),
        ('c1[c]cccc1', 'c1ccccc1'),
        ('C[NH2]', 'CN'),
        ('[CH2]C[CH2]', 'CCC')
    ])
@pytest.mark.parametrize('cheap', [(True,), (False,)])
def test_get_closed_shell_mol(rad_smi, expect_smi, cheap):

    rad_mol = RDKitMol.FromSmiles(rad_smi)
    assert rad_mol.GetClosedShellMol(cheap=cheap).ToSmiles() == expect_smi


def test_copy_method():
    """
    Test copy molecule functionality.
    """
    smi = "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]"
    mol = RDKitMol.FromSmiles(smi)
    mol.EmbedConformer()

    mol_copy = mol.Copy()
    assert mol.__hash__() != mol_copy.__hash__()
    assert mol.GetAtomicNumbers() == mol_copy.GetAtomicNumbers()
    assert mol.GetNumConformers() == mol_copy.GetNumConformers()
    assert np.allclose(mol.GetPositions(), mol_copy.GetPositions())

    mol_copy = mol.Copy(quickCopy=True)
    assert mol.__hash__() != mol_copy.__hash__()
    assert mol.GetAtomicNumbers() == mol_copy.GetAtomicNumbers()
    assert mol_copy.GetNumConformers() == 0

    mol_copy = mol.Copy()
    mol_copy.KeepIDs = {1: True, 2: False}
    mol_copy2 = mol_copy.Copy(copy_attrs=["KeepIDs"])

    assert hasattr(mol_copy2, "KeepIDs")
    assert mol_copy2.KeepIDs == mol_copy.KeepIDs


def test_get_torsion_tops():
    """
    Test get torsion tops of a given molecule.
    """
    smi1 = "[C:1]([C:2]([H:6])([H:7])[H:8])([H:3])([H:4])[H:5]"
    mol = RDKitMol.FromSmiles(smi1)
    tops = mol.GetTorsionTops([2, 0, 1, 5])
    assert len(tops) == 2
    assert set(tops) == {(0, 2, 3, 4), (1, 5, 6, 7)}

    smi2 = "[C:1]([C:2]#[C:3][C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]"
    mol = RDKitMol.FromSmiles(smi2)
    with pytest.raises(ValueError):
        mol.GetTorsionTops([4, 0, 3, 7])
    tops = mol.GetTorsionTops([4, 0, 3, 7], allowNonBondPivots=True)
    assert len(tops) == 2
    assert set(tops) == {(0, 4, 5, 6), (3, 7, 8, 9)}

    smi3 = "[C:1]([H:3])([H:4])([H:5])[H:6].[O:2][H:7]"
    mol = RDKitMol.FromSmiles(smi3)
    mol = mol.AddBonds([[1, 2]])
    with pytest.raises(ValueError):
        mol.GetTorsionTops([3, 0, 1, 6])
    tops = mol.GetTorsionTops([3, 0, 1, 6], allowNonBondPivots=True)
    assert len(tops) == 2
    assert set(tops) == {(0, 3, 4, 5), (1, 6)}


def test_add_bonds():
    """
    Test adding redundant bond to a molecule.
    """
    smi = "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]"

    # Add a redundant bond (C2-H3) that exists in the molecule
    # This should raise an error
    mol1 = RDKitMol.FromSmiles(smi)
    assert mol1.GetBondBetweenAtoms(1, 2) is not None
    with pytest.raises(RuntimeError):
        mol1.AddBonds([(1, 2)], inplace=False)

    # Add a bond between (H1-H3)
    mol2 = RDKitMol.FromSmiles(smi)
    mol_w_new_bond = mol2.AddBonds([(0, 2)], inplace=False)
    # mol_w_new_bond should be different mol objects
    assert mol_w_new_bond.__hash__() != mol2.__hash__()
    # The new mol should contain the new bond
    assert mol2.GetBondBetweenAtoms(0, 2) is None
    new_bond = mol_w_new_bond.GetBondBetweenAtoms(0, 2)
    assert new_bond is not None
    # The redundant bond has a bond order of 1.0
    assert new_bond.GetBondTypeAsDouble() == 1.0


def test_renumber_atoms():
    """
    Test the functionality of renumber atoms of a molecule.
    """
    # A molecule complex
    smi = (
        "[C:1]([C:2]([C:3]([H:20])([H:21])[H:22])([O:4])[C:5]([H:23])([H:24])[H:25])"
        "([H:17])([H:18])[H:19].[C:6]([C:7]([C:8]([H:29])([H:30])[H:31])([C:9]([H:32])"
        "([H:33])[H:34])[c:10]1[c:11]([H:35])[c:12]([H:36])[c:13]([O:14][H:37])[c:15]"
        "([H:38])[c:16]1[H:39])([H:26])([H:27])[H:28]"
    )

    # The generated molecule will maintain all the atom map numbers and the atom indexes
    # have the same sequence as the atom map numbers
    ref_mol = RDKitMol.FromSmiles(smi, keepAtomMap=True)

    # Since the molecule atom indexes are consistent with the atom map numbers
    # The generated molecule should have the same atom map numbers
    assert (
        ref_mol.RenumberAtoms(updateAtomMap=False).GetAtomMapNumbers()
        == ref_mol.GetAtomMapNumbers()
    )

    # Create a molecule with different atom indexes
    mols = [RDKitMol.FromSmiles(smi, keepAtomMap=True) for smi in smi.split(".")]
    combined = mols[0].CombineMol(mols[1])
    # If not renumbered, then atom maps and atom sequences are different
    assert combined.GetAtomMapNumbers() != ref_mol.GetAtomMapNumbers()
    assert combined.GetAtomicNumbers() != ref_mol.GetAtomicNumbers()
    # Atom maps and atom sequences are the same to the reference molecule now
    assert (
        combined.RenumberAtoms(updateAtomMap=False).GetAtomMapNumbers()
        == ref_mol.GetAtomMapNumbers()
    )
    assert (
        combined.RenumberAtoms(updateAtomMap=False).GetAtomicNumbers()
        == ref_mol.GetAtomicNumbers()
    )

    smi = "[C:1]([H:2])([H:3])([H:4])[H:5]"
    ref_mol = RDKitMol.FromSmiles(smi)
    # Renumber molecule but keep the original atom map
    renumbered = ref_mol.RenumberAtoms([1, 2, 3, 4, 0], updateAtomMap=False)
    assert renumbered.GetAtomMapNumbers() == (2, 3, 4, 5, 1)
    assert renumbered.GetAtomicNumbers() == [1, 1, 1, 1, 6]
    # Renumber molecule but also update the atom map after renumbering
    renumbered = ref_mol.RenumberAtoms([1, 2, 3, 4, 0], updateAtomMap=True)
    assert renumbered.GetAtomMapNumbers() == (1, 2, 3, 4, 5)
    assert renumbered.GetAtomicNumbers() == [1, 1, 1, 1, 6]

    "[C:1]([H:2])([H:3])([H:4])[H:5]"
    ref_mol = RDKitMol.FromSmiles(smi)
    # Renumber molecule with a dict
    renumbered = ref_mol.RenumberAtoms({0: 4, 4: 0}, updateAtomMap=True)
    assert renumbered.GetAtomMapNumbers() == (1, 2, 3, 4, 5)
    assert renumbered.GetAtomicNumbers() == [1, 1, 1, 1, 6]


def test_combined_mol():
    """
    Test combining molecules using CombineMol.
    """
    xyz_1 = np.array(
        [
            [-0.01841209, -0.00118705, 0.00757447],
            [-0.66894707, -0.81279485, -0.34820667],
            [-0.36500814, 1.00785186, -0.31659064],
            [0.08216461, -0.04465528, 1.09970299],
            [0.97020269, -0.14921467, -0.44248015],
        ]
    )
    xyz_2 = np.array([[0.49911347, 0.0, 0.0], [-0.49911347, 0.0, 0.0]])
    m1 = RDKitMol.FromSmiles("C")
    m1.EmbedConformer()
    m1.SetPositions(xyz_1)

    m2 = RDKitMol.FromSmiles("[OH]")
    m2.EmbedConformer()
    m2.SetPositions(xyz_2)

    combined = m1.CombineMol(m2)
    assert np.allclose(
        np.concatenate(
            [xyz_1, xyz_2],
        ),
        combined.GetPositions(),
    )

    combined = m1.CombineMol(m2, np.array([1.0, 1.0, 0.0]))
    assert np.allclose(
        np.concatenate(
            [xyz_1, xyz_2 + np.array([1.0, 1.0, 0.0])],
        ),
        combined.GetPositions(),
    )

    combined = m2.CombineMol(m1, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(
        np.concatenate(
            [xyz_2, xyz_1 + np.array([0.0, 0.0, 1.0])],
        ),
        combined.GetPositions(),
    )

    m1.EmbedMultipleConfs(10)
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
