import pytest
import numpy as np

from rdkit import Chem

try:
    # Openbabel 3
    from openbabel import openbabel as ob
except ImportError:
    # Openbabel 2
    import openbabel as ob


from rdmc.rdtools.obabel import (
    get_obmol_coords,
    openbabel_mol_to_rdkit_mol,
    parse_xyz_by_openbabel,
    rdkit_mol_to_openbabel_mol,
    rdkit_mol_to_openbabel_mol_manual,
    set_obmol_coords,
)

from rdmc.rdtools.conversion import mol_from_smiles
from rdmc.rdtools.element import get_atomic_num


@pytest.mark.parametrize(
    "xyz_str, xyz_coords",
    [
        (
            """9

C      0.877326   -0.098250   -0.084854
C     -0.485434    0.500788    0.077366
O     -1.467272   -0.400748   -0.379978
H      0.761119   -1.199281   -0.258052
H      1.505966    0.018086    0.835736
H      1.450469    0.313605   -0.935789
H     -0.650946    0.703458    1.166255
H     -0.645038    1.419756   -0.523925
H     -1.346190   -1.257412    0.103241""",
            np.array(
                [
                    [0.87732605, -0.09825022, -0.08485408],
                    [-0.48543355, 0.50078797, 0.0773663],
                    [-1.46727204, -0.40074832, -0.3799782],
                    [0.76111859, -1.19928127, -0.25805208],
                    [1.50596584, 0.01808555, 0.83573634],
                    [1.45046879, 0.3136052, -0.93578915],
                    [-0.65094569, 0.70345754, 1.16625452],
                    [-0.64503773, 1.41975555, -0.52392462],
                    [-1.34619025, -1.25741199, 0.10324098],
                ]
            ),
        )
    ],
)
def test_get_obmol_coords(xyz_str, xyz_coords):
    obconv, obmol = ob.OBConversion(), ob.OBMol()
    obconv.SetInFormat("xyz")
    obconv.ReadString(obmol, xyz_str)

    np.testing.assert_array_almost_equal(
        get_obmol_coords(obmol),
        xyz_coords,
        decimal=6,
    )


@pytest.mark.parametrize(
    "xyz, smi, mult",
    [
        (
            """8

C     -0.763437    0.050567   -0.027960
C      0.735495   -0.000842   -0.048377
O      1.230196   -0.419073    1.194244
H     -1.176457    1.066091   -0.162871
H     -1.204191   -0.579978   -0.825262
H     -1.138725   -0.301251    0.965053
H      1.219517    0.947190   -0.311463
H      1.097602   -0.762704   -0.783363""",
            "CC[O]",
            2,
        ),
        (
            """6

C     -0.387408    0.000432    0.000429
N      0.944508   -0.436568    0.409251
H     -0.855579   -0.809884   -0.612157
H     -0.318201    0.948418   -0.574669
H     -1.021874    0.176059    0.891897
H      1.638555    0.121543   -0.114750""",
            "C[NH]",
            2,
        ),
        (
            """4

C     -0.003054    0.012107    0.249028
H      1.038218   -0.178411   -0.082059
H     -0.642509   -0.823783   -0.081409
H     -0.392655    0.990087   -0.085560""",
            "[CH3]",
            2,
        ),
        (
            """5

C      0.001156   -0.019902   -0.015705
H     -0.813529   -0.726520   -0.214171
H     -0.307828    0.982159   -0.365147
H      0.163077    0.057280    1.071021
H      0.957126   -0.293016   -0.475997""",
            "C",
            1,
        ),
        (
            """1

Cl    0.000000    0.000000    0.000000""",
            "[Cl]",
            2,
        ),
    ],
)
def test_parse_xyz_by_openbabel(xyz, smi, mult):
    obmol = parse_xyz_by_openbabel(xyz)

    # Check molecule properties are correct
    num_atoms = int(xyz.splitlines()[0].strip())
    assert obmol.NumAtoms() == num_atoms
    assert [obmol.GetAtomById(i).GetAtomicNum() for i in range(num_atoms)] == [
        get_atomic_num(atom.strip().split()[0])
        for atom in xyz.splitlines()[2 : 2 + num_atoms]
    ]
    assert obmol.GetTotalSpinMultiplicity() == mult

    # Check if the coordinates are correctly converted
    xyz_coords = get_obmol_coords(obmol)
    assert len(xyz_coords) == num_atoms
    np.testing.assert_array_almost_equal(
        xyz_coords,
        np.array(
            [
                [
                    float(coord_values)
                    for coord_values in coord_values.strip().split()[1:]
                ]
                for coord_values in xyz.splitlines()[2 : 2 + num_atoms]
            ]
        ),
        decimal=6,
    )

    # Check if the molecule is correct
    obconv = ob.OBConversion()
    obconv.SetOutFormat("smi")
    assert obconv.WriteString(obmol).strip() == smi


@pytest.mark.parametrize(
    "smi, xyz",
    [
        (
            "CC[O]",
            np.array(
                [
                    [-0.763437, 0.050567, -0.027960],
                    [0.735495, -0.000842, -0.048377],
                    [1.230196, -0.419073, 1.194244],
                ]
            ),
        ),
        ("C", np.array([[0.001156, -0.019902, -0.015705]])),
        ("[Cl]", np.array([[0.0, 0.0, 0.0]])),
    ],
)
def test_set_obmol_coords(smi, xyz):
    obconv, obmol = ob.OBConversion(), ob.OBMol()
    obconv.SetInFormat("smi")
    obconv.ReadString(obmol, smi)

    set_obmol_coords(obmol, xyz)

    assert get_obmol_coords(obmol).shape == xyz.shape
    np.testing.assert_array_almost_equal(
        get_obmol_coords(obmol),
        xyz,
        decimal=6,
    )


@pytest.mark.parametrize(
    "smi",
    [
        "CC[O]",
        "C[NH]",
        "[CH3]",
        "C",
    ],
)
@pytest.mark.parametrize("embed", [True, False])
@pytest.mark.parametrize("add_hs", [True, False])
def test_rdkit_mol_to_openbabel_mol(smi, add_hs, embed):
    mol = mol_from_smiles(smi, add_hs=add_hs)
    if embed:
        Chem.AllChem.EmbedMolecule(mol)

    obmol = rdkit_mol_to_openbabel_mol(mol, embed=embed)
    assert obmol.NumAtoms() == mol.GetNumAtoms()
    obconv = ob.OBConversion()
    obconv.SetOutFormat("smi")
    assert obconv.WriteString(obmol).strip() == smi

    if embed:
        np.testing.assert_array_almost_equal(
            get_obmol_coords(obmol),
            mol.GetConformer().GetPositions(),
            decimal=4,  # limited by RDKit MolToMolBlock
        )


@pytest.mark.parametrize(
    "smi",
    [
        "CC[O]",
        "C[NH]",
        "[CH3]",
        "C",
    ],
)
@pytest.mark.parametrize("embed", [True, False])
@pytest.mark.parametrize("add_hs", [True, False])
def test_rdkit_mol_to_openbabel_mol_manual(smi, add_hs, embed):
    mol = mol_from_smiles(smi, add_hs=add_hs)
    if embed:
        Chem.AllChem.EmbedMolecule(mol)

    obmol = rdkit_mol_to_openbabel_mol_manual(mol, embed=embed)
    assert obmol.NumAtoms() == mol.GetNumAtoms()
    obconv = ob.OBConversion()
    obconv.SetOutFormat("smi")
    assert obconv.WriteString(obmol).strip() == smi

    if embed:
        np.testing.assert_array_almost_equal(
            get_obmol_coords(obmol),
            mol.GetConformer().GetPositions(),
            decimal=6,
        )


@pytest.mark.parametrize(
    "xyz, smi, mult",
    [
        (
            """8

C     -0.763437    0.050567   -0.027960
C      0.735495   -0.000842   -0.048377
O      1.230196   -0.419073    1.194244
H     -1.176457    1.066091   -0.162871
H     -1.204191   -0.579978   -0.825262
H     -1.138725   -0.301251    0.965053
H      1.219517    0.947190   -0.311463
H      1.097602   -0.762704   -0.783363""",
            "CC[O]",
            2,
        ),
        (
            """6

C     -0.387408    0.000432    0.000429
N      0.944508   -0.436568    0.409251
H     -0.855579   -0.809884   -0.612157
H     -0.318201    0.948418   -0.574669
H     -1.021874    0.176059    0.891897
H      1.638555    0.121543   -0.114750""",
            "C[NH]",
            2,
        ),
        (
            """4

C     -0.003054    0.012107    0.249028
H      1.038218   -0.178411   -0.082059
H     -0.642509   -0.823783   -0.081409
H     -0.392655    0.990087   -0.085560""",
            "[CH3]",
            2,
        ),
        (
            """5

C      0.001156   -0.019902   -0.015705
H     -0.813529   -0.726520   -0.214171
H     -0.307828    0.982159   -0.365147
H      0.163077    0.057280    1.071021
H      0.957126   -0.293016   -0.475997""",
            "C",
            1,
        ),
        (
            """1

Cl    0.000000    0.000000    0.000000""",
            "[Cl]",
            2,
        ),
    ],
)
@pytest.mark.parametrize("remove_hs", [True, False])
@pytest.mark.parametrize("sanitize", [True, False])
@pytest.mark.parametrize("embed", [True, False])
def test_openbabel_mol_to_rdkit_mol(xyz, smi, mult, remove_hs, sanitize, embed):
    obmol = parse_xyz_by_openbabel(xyz)
    mol = openbabel_mol_to_rdkit_mol(
        obmol,
        remove_hs=remove_hs,
        sanitize=sanitize,
        embed=embed,
    )
    if remove_hs:
        assert mol.GetNumAtoms() == obmol.NumHvyAtoms()
    else:
        assert mol.GetNumAtoms() == obmol.NumAtoms()
    assert sum([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()]) + 1 == mult
    assert mol.GetNumConformers() == 0 if not embed else 1

    if embed and not remove_hs:
        np.testing.assert_array_almost_equal(
            mol.GetConformer().GetPositions(),
            get_obmol_coords(obmol),
            decimal=6,
        )
    elif embed:
        heavy_atom_idxs = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
        ]
        np.testing.assert_array_almost_equal(
            mol.GetConformer().GetPositions(),
            get_obmol_coords(obmol)[heavy_atom_idxs],
            decimal=6,
        )

    # Need to remove Hs for SMILES comparison
    mol = Chem.rdmolops.RemoveHs(mol, sanitize=True)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == smi
