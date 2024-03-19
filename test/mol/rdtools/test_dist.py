import pytest

from rdkit import Chem

from rdmc.mol import RDKitMol
from rdmc.rdtools.dist import get_missing_bonds, has_colliding_atoms


@pytest.mark.parametrize(
    "xyz, threshold, expect_bonds",
    [
        # A case of H abstraction of transition state
        (
            """C          0.86917        1.50073       -0.41702
C         -0.47607        2.03752        0.01854
H          0.85087        0.99684       -1.38520
H          1.16254        0.53083        0.35265
C         -1.66031        1.18407       -0.41323
H         -0.48474        2.19298        1.10199
H         -0.61511        3.02720       -0.42857
N         -1.64328       -0.19909        0.05343
H         -1.68450        1.15374       -1.50571
H         -2.58945        1.68138       -0.08673
C         -2.74796       -0.93786       -0.53898
C         -1.67887       -0.29955        1.50384
H         -2.69951       -1.98186       -0.22875
H         -3.72671       -0.52994       -0.24178
H         -2.67337       -0.90452       -1.62638
H         -1.69925       -1.35091        1.79085
H         -2.56817        0.19427        1.92693
H         -0.78293        0.13647        1.93981
O          1.61241       -0.39643        1.10878
O          2.26285       -1.25639        0.24798
C          1.32762       -2.20908       -0.22666
O          0.43662       -1.69114       -1.13896
H          1.94625       -2.95523       -0.72774
H          0.82192       -2.64082        0.64549
H         -0.24063       -1.15943       -0.66324
N          1.91323        2.44655       -0.35205
H          1.98740        2.90733        0.54373
H          2.80917        2.06603       -0.61836""",
            1.5,
            [(3, 18)],
        ),
        # A case of oxonium species
        (
            """O     -1.2607590000    0.7772420000    0.6085820000
C     -0.1650470000   -2.3539430000    2.2668210000
C     -0.4670120000   -2.1947580000    0.7809780000
C      0.5724080000   -1.3963940000   -0.0563730000
C      1.9166170000   -2.1487680000   -0.0973880000
C      0.0355110000   -1.2164630000   -1.4811920000
C      0.8592950000   -0.0701790000    0.6147050000
O      1.6293140000    0.1954080000    1.4668300000
O      0.0710230000    1.0551410000    0.0304340000
C      0.5008030000    2.3927920000    0.4116770000
H     -0.9212150000   -2.9917470000    2.7288580000
H     -0.1856660000   -1.3928280000    2.7821170000
H      0.8077150000   -2.8148360000    2.4472520000
H     -1.4311160000   -1.7082160000    0.6552790000
H     -0.5276310000   -3.1794610000    0.3074300000
H      1.7489410000   -3.1449730000   -0.5091360000
H      2.3570430000   -2.2480580000    0.8923780000
H      2.6337360000   -1.6301710000   -0.7383130000
H     -0.0590770000   -2.2002990000   -1.9397630000
H      0.7068050000   -0.6180050000   -2.0971060000
H     -0.9435140000   -0.7413070000   -1.4727710000
H      0.4382590000    2.4894460000    1.4913270000
H     -0.1807540000    3.0525390000   -0.1120870000
H      1.5196670000    2.5089310000    0.0492140000""",
            1.5,
            [(6, 8)],
        ),
    ],
)
def test_missing_bonds(xyz, threshold, expect_bonds):
    mol = RDKitMol.FromXYZ(xyz, header=False)

    missing_bonds = get_missing_bonds(mol, threshold=threshold)

    assert missing_bonds == expect_bonds


@pytest.mark.parametrize("reference", ["vdw", "covalent", "bond"])
def test_has_colliding_atoms_false(reference):
    # RDKit ETKDG methods won't results in colliding atoms
    mol = RDKitMol.FromSmiles("CC")

    Chem.AllChem.EmbedMolecule(mol)

    assert not has_colliding_atoms(mol, reference=reference)


def test_has_colliding_atoms_true():
    # Create a [CH3].[H] molecule
    mol = RDKitMol.FromSmiles("[C:1]([H:2])([H:3])[H:4].[H:5]")
    # When embedding, the H atom is placed at the origin, causing colliding atoms
    Chem.AllChem.EmbedMolecule(mol)

    assert has_colliding_atoms(mol, reference="vdw")
