#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import pytest

from rdmc import RDKitMol
from rdmc.fix import ALL_REMEDIES, fix_mol


@pytest.mark.parametrize(
    "smi, exp_smi",
    [
        ("O=O", "[O][O]"),
        ("[C]=O", "[C-]#[O+]"),
        ("CS(C)([O])[O]", "CS(C)(=O)=O"),
        ("[CH2]O[O]", "[CH2+]O[O-]"),
        ("[CH2]C=CO[O]", "C=C[CH+]O[O-]"),
    ],
)
def test_fix_sanitize_ok(smi, exp_smi):
    mol = RDKitMol.FromSmiles(smi)
    assert fix_mol(mol, ALL_REMEDIES).ToSmiles() == exp_smi


@pytest.mark.parametrize(
    "smi, exp_smi",
    [
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
