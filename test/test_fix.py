#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import pytest

from rdmc import RDKitMol
from rdmc.fix import ALL_REMEDIES, fix_mol, fix_oxonium_bonds


@pytest.mark.parametrize(
    "smi, exp_smi",
    [
        ("O=O", "[O][O]"),
        ("[C]=O", "[C-]#[O+]"),
        ("CS(C)([O])[O]", "CS(C)(=O)=O"),
        ("[CH2]O[O]", "C=[O+][O-]"),
        ("[CH2]C=CO[O]", "C=CC=[O+][O-]"),
    ],
)
def test_fix_sanitize_ok(smi, exp_smi):
    mol = RDKitMol.FromSmiles(smi)
    assert fix_mol(mol, ALL_REMEDIES).ToSmiles() == exp_smi


@pytest.mark.parametrize(
    "smi, exp_smi",
    [
        ("[C]#[O]", "[C-]#[O+]"),
        ("[NH3][O]", "[NH3+][O-]"),
        ("[CH2][NH3]", "[CH2-][NH3+]"),
        ("[C]#[NH]", "[C-]#[NH+]"),
        ("[N]=N=N", "[N-]=[N+]=N"),
        ("[CH2]=[NH2]", "[CH2]N"),
        ("O=[CH]=O", "[O]C=O"),
        ("[CH]1=N=[CH]=N=[CH]=N=1", "c1ncncn1"),
        ("[NH3][CH]=O", "[NH3+][CH][O-]"),
        ("[NH3][CH][O]", "[NH3+][CH][O-]"),
        ("[NH3]CC(=O)[O]", "[NH3+]CC(=O)[O-]"),
        ("[NH3]CS(=O)(=O)[O]", "[NH3+]CS(=O)(=O)[O-]"),
        ("[NH3]CP(=O)([O])O", "[NH3+]CP(=O)([O-])O"),
    ],
)
def test_fix_sanitize_bad_non_resonance(smi, exp_smi):
    mol = RDKitMol.FromSmiles(smi, sanitize=False)
    with pytest.raises(Exception):
        mol.Sanitize()
    assert fix_mol(mol, ALL_REMEDIES).ToSmiles() == exp_smi


def test_fix_mol_complex():
    mol = RDKitMol.FromSmiles("O=O.[C]=O")
    assert set(fix_mol(mol, ALL_REMEDIES).ToSmiles().split(".")) == set(
        ["[O][O]", "[C-]#[O+]"]
    )


def test_fix_spin_multiplicity():
    mol = RDKitMol.FromSmiles("[CH2][CH2]")
    assert fix_mol(mol, fix_spin_multiplicity=True, mult=1).GetSpinMultiplicity() == 1


def test_renumber_after_fix():
    mol = RDKitMol.FromSmiles("[H:1][C:2]([H:3])[N:4]#[C:5]", sanitize=False)
    mol_fix = fix_mol(mol.Copy(quickCopy=True), renumber_atoms=False)
    assert mol.GetAtomMapNumbers() != mol_fix.GetAtomMapNumbers()
    mol_fix = fix_mol(mol.Copy(quickCopy=True), renumber_atoms=True)
    assert mol.GetAtomMapNumbers() == mol_fix.GetAtomMapNumbers()
    assert mol.GetAtomicNumbers() == mol_fix.GetAtomicNumbers()


@pytest.mark.parametrize(
    "xyz, exp_smi",
    [
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
            "CCC(C)(C)C(=O)[O+](C)[O-]",
        ),
        (
            """O       -1.17118700      -1.50146090      -2.47413980
H       -2.01769360      -1.92927720      -2.63341740
C       -1.98485810       2.24098580      -2.21737070
C       -1.70946220       0.80773220      -1.77051860
C       -1.32477280      -0.00677400      -2.94764940
H       -0.87667130      -1.54124320      -1.46319570
H       -0.30275440       0.21066780      -3.26524910
N       -0.71008250       0.93599160      -0.58681770
C       -0.18358680      -0.06896500       0.08764570
O       -0.33148990      -1.27959420      -0.16793930
H       -2.82713830       2.25060270      -2.90397100
H       -1.11829350       2.64922810      -2.74266840
H       -2.21700900       2.89116810      -1.36932880
H       -2.60626880       0.39545810      -1.29598380
H       -0.49062240       1.86220920      -0.24896060
H        0.43540870       0.21959280       0.94605120""",
            "CC([CH-][OH2+])NC=O",
        ),
    ],
)
def test_fix_oxonium(xyz, exp_smi):
    mol = RDKitMol.FromXYZ(xyz, sanitize=False, header=False)
    assert fix_oxonium_bonds(mol).ToSmiles() == exp_smi
