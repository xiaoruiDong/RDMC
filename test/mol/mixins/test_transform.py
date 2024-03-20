import pytest
from rdmc import RDKitMol

try:
    from openbabel import pybel
except ImportError:
    import pybel

@pytest.mark.parametrize(
    "inchi",
    [
        "InChI=1S/H2O/h1H2",
        "InChI=1S/CH3/h1H3",
        "InChI=1S/H2O/h1H2/p-1",
        "InChI=1S/H3N/h1H3/p+1",
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
    ],
)
def test_mol_to_inchi(inchi):
    """
    Test converting RDKitMol to InChI strings.
    """
    assert inchi == RDKitMol.FromInchi(inchi).ToInchi()


def test_generate_mol_from_inchi():
    """
    Test generate the RDKitMol from InChI strings.
    """
    # InChI of a stable species
    inchi1 = "InChI=1S/H2O/h1H2"
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
    inchi2 = "InChI=1S/CH3/h1H3"
    mol3 = RDKitMol.FromInchi(inchi2)
    assert mol3.GetNumAtoms() == 4
    assert mol3.GetAtomicNumbers() == [6, 1, 1, 1]
    assert mol3.GetAtomWithIdx(0).GetNumRadicalElectrons() == 1
    assert set(mol3.GetBondsAsTuples()) == {(0, 1), (0, 2), (0, 3)}

    # InChI of an anion
    inchi3 = "InChI=1S/H2O/h1H2/p-1"
    mol4 = RDKitMol.FromInchi(inchi3)
    assert mol4.GetNumAtoms() == 2
    assert mol4.GetAtomicNumbers() == [8, 1]
    assert mol4.GetAtomWithIdx(0).GetFormalCharge() == -1
    assert set(mol4.GetBondsAsTuples()) == {(0, 1)}

    # InChI of an cation
    inchi4 = "InChI=1S/H3N/h1H3/p+1"
    mol5 = RDKitMol.FromInchi(inchi4)
    assert mol5.GetNumAtoms() == 5
    assert mol5.GetAtomicNumbers() == [7, 1, 1, 1, 1]
    assert mol5.GetAtomWithIdx(0).GetFormalCharge() == 1
    assert set(mol5.GetBondsAsTuples()) == {(0, 1), (0, 2), (0, 3), (0, 4)}

    # InChI of benzene (aromatic ring)
    inchi5 = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    mol6 = RDKitMol.FromInchi(inchi5)
    assert len(mol6.GetAromaticAtoms()) == 6


def test_mol_to_xyz():
    """
    Test converting RDKitMol to XYZ strings.
    """
    xyz = """1\n\nH      0.000000    0.000000    0.000000\n"""
    mol = RDKitMol.FromXYZ(xyz)
    assert mol.ToXYZ(header=True) == xyz
    assert (
        mol.ToXYZ(header=False) == xyz[3:]
    )  # Currently to XYZ without header has no line break at the end
    assert (
        mol.ToXYZ(header=True, comment="test")
        == """1\ntest\nH      0.000000    0.000000    0.000000\n"""
    )


@pytest.mark.parametrize(
    "smi",
    [
        "[C-]#[O+]",
        "[C]",
        "[CH]",
        "OO",
        "[H][H]",
        "[H]",
        "[He]",
        "[O]",
        "O",
        "[CH3]",
        "C",
        "[OH]",
        "CCC",
        "CC",
        "N#N",
        "[O]O",
        "[CH2]C",
        "[Ar]",
        "CCCC",
        "O=C=O",
        "[C]#N",
    ],
)
def test_smiles_without_atom_mapping_and_hs(smi):
    """
    Test exporting a molecule as a SMILES string without atom mapping and explicit H atoms.
    """
    assert RDKitMol.FromSmiles(smi).ToSmiles() == smi


def test_smiles_with_atom_mapping_and_hs():
    """
    Test exporting a molecule as a SMILES string with atom mapping and explicit H atoms.
    """
    # Normal SMILES without atom mapping, atommap and H atoms will be
    # assigned during initiation
    mol1 = RDKitMol.FromSmiles("[CH2]C", assignAtomMap=True)
    # Export SMILES with H atoms
    assert (
        mol1.ToSmiles(
            removeHs=False,
        )
        == "[H][C]([H])C([H])([H])[H]"
    )
    # Export SMILES with H atoms and indexes
    assert (
        mol1.ToSmiles(removeHs=False, removeAtomMap=False)
        == "[C:1]([C:2]([H:5])([H:6])[H:7])([H:3])[H:4]"
    )

    # SMILES with atom mapping
    mol2 = RDKitMol.FromSmiles("[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]")
    # Test the atom indexes and atom map numbers share the same order
    assert mol2.GetAtomMapNumbers() == (1, 2, 3, 4, 5, 6, 7)
    # Test the 2nd and 4th atoms are carbons
    assert mol2.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol2.GetAtomWithIdx(3).GetAtomicNum() == 6
    # Export SMILES without H atoms and atom map
    assert mol2.ToSmiles() == "[CH2]C"
    # Export SMILES with H atoms and without atom map
    assert (
        mol2.ToSmiles(
            removeHs=False,
        )
        == "[H][C]([H])C([H])([H])[H]"
    )
    # Export SMILES without H atoms and with atom map
    # Atom map numbers for heavy atoms are perserved
    assert (
        mol2.ToSmiles(
            removeAtomMap=False,
        )
        == "[CH3:2][CH2:4]"
    )
    # Export SMILES with H atoms and with atom map
    assert (
        mol2.ToSmiles(
            removeHs=False,
            removeAtomMap=False,
        )
        == "[H:1][C:4]([C:2]([H:5])([H:6])[H:7])[H:3]"
    )

    # SMILES with atom mapping but neglect the atom mapping
    mol3 = RDKitMol.FromSmiles(
        "[H:6][C:2]([C:4]([H:1])[H:3])([H:5])[H:7]", keepAtomMap=False
    )
    # Test the atom indexes and atom map numbers share the same order
    assert mol3.GetAtomMapNumbers() == (1, 2, 3, 4, 5, 6, 7)
    # However, now the 4th atom is not carbon (3rd instead), and atom map numbers
    # are determined by the sequence of atom appear in the SMILES.
    assert mol3.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol3.GetAtomWithIdx(2).GetAtomicNum() == 6
    # Export SMILES with H atoms and with atom map
    assert (
        mol3.ToSmiles(
            removeHs=False,
            removeAtomMap=False,
        )
        == "[H:1][C:2]([C:3]([H:4])[H:5])([H:6])[H:7]"
    )

    # SMILES with uncommon atom mapping starting from 0 and being discontinue
    mol4 = RDKitMol.FromSmiles("[H:9][C:2]([C:5]([H:0])[H:3])([H:4])[H:8]")
    # Test the atom indexes and atom map numbers share the same order
    assert mol4.GetAtomMapNumbers() == (0, 2, 3, 4, 5, 8, 9)
    # Check Heavy atoms' index
    assert mol4.GetAtomWithIdx(1).GetAtomicNum() == 6
    assert mol4.GetAtomWithIdx(4).GetAtomicNum() == 6
    # Export SMILES without H atoms and with atom map
    # Atom map numbers for heavy atoms are perserved
    assert (
        mol4.ToSmiles(
            removeAtomMap=False,
        )
        == "[CH3:2][CH2:5]"
    )
    # Export SMILES with H atoms and with atom map
    assert (
        mol4.ToSmiles(removeHs=False, removeAtomMap=False)
        == "[H:0][C:5]([C:2]([H:4])([H:8])[H:9])[H:3]"
    )

def test_generate_mol_from_openbabel_mol():
    """
    Test generating the RDKitMol object from an Openbabel Molecule object.
    """
    # Generate from openbabel without embedded geometries
    pmol = pybel.readstring("smi", "CC")
    pmol.addh()
    ob_mol = pmol.OBMol
    mol1 = RDKitMol.FromOBMol(ob_mol)  # default arguments
    assert mol1.ToSmiles() == "CC"
    assert mol1.GetNumConformers() == 0

    # Generate from OBMol with geometries
    pmol = pybel.readstring("smi", "CCC")
    pmol.addh()
    pmol.make3D()
    ob_mol = pmol.OBMol
    mol2 = RDKitMol.FromOBMol(ob_mol)
    assert mol2.ToSmiles() == "CCC"
    assert mol2.GetNumConformers() == 1
    assert mol2.GetPositions().shape == (11, 3)
