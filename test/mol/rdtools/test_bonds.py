import pytest

from rdkit import Chem

from rdmc.rdtools.bond import (
    add_bond,
    add_bonds,
    get_bonds_as_tuples,
)

from rdmc.mol import RDKitMol


def test_add_bond():
    """
    Test the function ``add_bond``.
    """
    mol0 = RDKitMol()
    mol0.AddAtom(Chem.Atom(6))
    mol0.AddAtom(Chem.Atom(6))
    assert len(mol0.GetBonds()) == 0
    mol1 = add_bond(mol0, (0, 1), inplace=False, update_properties=False)
    assert len(mol0.GetBonds()) == 0
    assert len(mol1.GetBonds()) == 1
    assert mol1.NeedsUpdatePropertyCache()
    assert mol1.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE

    mol1.AddAtom(Chem.Atom(6))
    mol2 = add_bond(mol1, (1, 2), Chem.BondType.DOUBLE)
    assert len(mol2.GetBonds()) == len(mol1.GetBonds()) == 2


def test_add_bonds():
    """
    Test the function ``add_bonds``.
    """
    mol0 = RDKitMol()
    for _ in range(3):
        mol0.AddAtom(Chem.Atom(6))
    add_bonds(mol0, [(0, 1), (1, 2), (0, 2)], update_properties=False)
    assert len(mol0.GetBonds()) == 3


@pytest.mark.parametrize(
    "smi, exp_bonds",
    [
        (
            "[O:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]",
            [(0, 1), (1, 2), (1, 5), (1, 6), (2, 3), (2, 4)],
        ),
        (
            "[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]",
            [(1, 2), (1, 3), (1, 4), (1, 5)],
        ),
    ],
)
def test_get_bond_as_tuples(smi, exp_bonds):
    """
    Test getBondsAsTuples returns a list of atom pairs corresponding to each bond.
    """
    # Single molecule
    mol = RDKitMol.FromSmiles(smi)
    bonds = get_bonds_as_tuples(mol)
    assert set(bonds) == set(exp_bonds)
