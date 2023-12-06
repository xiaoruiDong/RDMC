import pytest

import numpy as np

from rdkit import Chem

from rdmc.rdtools.element import get_atomic_num
from rdmc.rdtools.xyz import mol_from_xyz

smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


mols = {
    "CC[O]": """8

C     -0.763437    0.050567   -0.027960
C      0.735495   -0.000842   -0.048377
O      1.230196   -0.419073    1.194244
H     -1.176457    1.066091   -0.162871
H     -1.204191   -0.579978   -0.825262
H     -1.138725   -0.301251    0.965053
H      1.219517    0.947190   -0.311463
H      1.097602   -0.762704   -0.783363""",
    "C[NH]": """6

C     -0.387408    0.000432    0.000429
N      0.944508   -0.436568    0.409251
H     -0.855579   -0.809884   -0.612157
H     -0.318201    0.948418   -0.574669
H     -1.021874    0.176059    0.891897
H      1.638555    0.121543   -0.114750""",
    "[CH3]": """4

C     -0.003054    0.012107    0.249028
H      1.038218   -0.178411   -0.082059
H     -0.642509   -0.823783   -0.081409
H     -0.392655    0.990087   -0.085560""",
    "C": """5

C      0.001156   -0.019902   -0.015705
H     -0.813529   -0.726520   -0.214171
H     -0.307828    0.982159   -0.365147
H      0.163077    0.057280    1.071021
H      0.957126   -0.293016   -0.475997""",
    "[Cl]": """1

Cl    0.000000    0.000000    0.000000""",
}


@pytest.mark.parametrize(
    "smi, xyz",
    mols.items(),
)
@pytest.mark.parametrize("header", [True, False])
def test_mol_from_xyz_openbabel(xyz, smi, header):
    xyz_wo_header = "\n".join(xyz.splitlines()[2:])
    num_atoms = int(xyz.splitlines()[0].strip())
    if not header:
        xyz = xyz_wo_header

    mol = mol_from_xyz(xyz, backend="openbabel", header=header)

    # Check molecule properties are correct
    assert mol.GetNumAtoms() == num_atoms
    assert [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)] == [
        get_atomic_num(atom.strip().split()[0]) for atom in xyz_wo_header.splitlines()
    ]

    # Check if the coordinates are correctly converted
    xyz_coords = mol.GetConformer().GetPositions()
    assert len(xyz_coords) == num_atoms
    np.testing.assert_array_almost_equal(
        xyz_coords,
        np.array(
            [
                [
                    float(coord_values)
                    for coord_values in coord_values.strip().split()[1:]
                ]
                for coord_values in xyz_wo_header.splitlines()
            ]
        ),
        decimal=6,
    )

    # Need to remove Hs for SMILES comparison
    mol = Chem.rdmolops.RemoveHs(mol, sanitize=True)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == smi


@pytest.mark.parametrize(
    "smi, xyz",
    mols.items(),
)
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("force_rdmc", [True, False])
def test_mol_from_xyz_xyz2mol(xyz, smi, header, force_rdmc):
    xyz_wo_header = "\n".join(xyz.splitlines()[2:])
    num_atoms = int(xyz.splitlines()[0].strip())
    if not header:
        xyz = xyz_wo_header

    mol = mol_from_xyz(xyz, backend="xyz2mol", header=header, force_rdmc=force_rdmc)

    # Check molecule properties are correct
    assert mol.GetNumAtoms() == num_atoms
    assert [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)] == [
        get_atomic_num(atom.strip().split()[0]) for atom in xyz_wo_header.splitlines()
    ]

    # Check if the coordinates are correctly converted
    xyz_coords = mol.GetConformer().GetPositions()
    assert len(xyz_coords) == num_atoms
    np.testing.assert_array_almost_equal(
        xyz_coords,
        np.array(
            [
                [
                    float(coord_values)
                    for coord_values in coord_values.strip().split()[1:]
                ]
                for coord_values in xyz_wo_header.splitlines()
            ]
        ),
        decimal=6,
    )

    # Need to remove Hs for SMILES comparison
    mol = Chem.rdmolops.RemoveHs(mol, sanitize=True)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == smi


@pytest.mark.parametrize("header", [True, False])
def test_invalid_backend(header):
    with pytest.raises(NotImplementedError):
        mol_from_xyz("C 0 0 0", backend="invalid", header=header)


@pytest.mark.parametrize("backend", ["openbabel", "xyz2mol"])
def test_invalid_xyz(backend):
    with pytest.raises(ValueError):
        mol_from_xyz("C 0 0 0", backend=backend, header=True)

@pytest.mark.parametrize(
    "xyz, chiral_tag",
    [
        (
            """9

C     -0.520349    0.276596   -0.074152
C      0.963111   -0.026999   -0.014479
Cl    -0.881814    1.852422    0.625563
O     -1.244240   -0.688146    0.604393
H     -0.793577    0.271512   -1.161074
H      1.470871    0.553919   -0.796426
H      1.172542   -1.095243   -0.150542
H      1.331785    0.257192    0.985589
H     -1.498328   -1.401254   -0.018873""",
            Chem.CHI_TETRAHEDRAL_CCW,
        ),
        (
            """5

C     -0.063472    0.129577    0.083450
F     -0.198158    1.208448   -0.782377
Cl    -1.251554   -1.145454   -0.272890
Br     1.697832   -0.646080   -0.165647
H     -0.184647    0.453509    1.137464""",
            Chem.CHI_TETRAHEDRAL_CW,
        )
    ]
)
@pytest.mark.parametrize("backend", ["openbabel", "xyz2mol"])
def test_embed_chiral(xyz, chiral_tag, backend):

    mol = mol_from_xyz(xyz, backend=backend, header=True, embed_chiral=True)

    assert mol.GetAtomWithIdx(0).GetChiralTag() == chiral_tag
