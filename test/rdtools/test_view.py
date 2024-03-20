"""Test viewers"""

# We can't really test if the generated 3Dmol.js object is all good. We only test if the workflow is error free.

import random

from rdkit import Chem
import py3Dmol
import pytest


from rdmc.rdtools.view import (
    mol_viewer,
    mol_animation,
    freq_viewer,
    conformer_viewer,
    conformer_animation,
    merge_xyz_dxdydz,
    ts_viewer,
    reaction_viewer,
    grid_viewer,
)


from rdmc.rdtools.view.utils import get_bonds_as_tuples, get_broken_formed_bonds, clean_ts


@pytest.fixture
def xyz():
    return (
        """2

    C   0.0     0.0     0.0
    H   0.0     0.0     0.8
    """
    )


@pytest.fixture
def dxdydz():
    return [[0.1, 0.0, -0.2], [0.2, 0.1, 0.3]]


def test_merge_xyz_dxdydz(xyz, dxdydz):

    expect = (
        "2\n\n"
        "C   0.0     0.0     0.0\t\t0.1         0.0         -0.2        \n"
        "H   0.0     0.0     0.8\t\t0.2         0.1         0.3         "
    )

    assert expect == merge_xyz_dxdydz(xyz, dxdydz)


smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


@pytest.fixture
def rmol():
    return Chem.MolFromSmiles("[H].[H][C][H]", smi_params)


@pytest.fixture
def pmol():
    return Chem.MolFromSmiles("[H][H].[C][H]", smi_params)


@pytest.fixture
def tsmol(rmol):
    return Chem.RWMol(rmol)


def get_mol_with_conformer(smi, n_confs=10):
    mol = Chem.MolFromSmiles(smi, smi_params)
    Chem.AllChem.EmbedMultipleConfs(mol, n_confs)
    return mol


def test_get_broken_formed_bonds(rmol, pmol):

    assert get_broken_formed_bonds(rmol, pmol) == ([(1, 2)], [(0, 1)])
    assert get_broken_formed_bonds(pmol, rmol) == ([(0, 1)], [(1, 2)])


def test_clean_ts(tsmol):

    cleaned_ts_mol = clean_ts(tsmol, [(1, 2)], [])
    assert cleaned_ts_mol.GetNumBonds() == 1
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(2, 3)]

    cleaned_ts_mol = clean_ts(tsmol, [], [(0, 1)])
    assert cleaned_ts_mol.GetNumBonds() == 2
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(1, 2), (2, 3)]

    cleaned_ts_mol = clean_ts(tsmol, [(1, 2)], [(0, 1)])
    assert cleaned_ts_mol.GetNumBonds() == 1
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(2, 3)]


def test_clean_ts_mol_object(rmol):

    cleaned_ts_mol = clean_ts(rmol, [(1, 2)], [])
    assert cleaned_ts_mol.GetNumBonds() == 1
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(2, 3)]

    cleaned_ts_mol = clean_ts(rmol, [], [(0, 1)])
    assert cleaned_ts_mol.GetNumBonds() == 2
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(1, 2), (2, 3)]

    cleaned_ts_mol = clean_ts(rmol, [(1, 2)], [(0, 1)])
    assert cleaned_ts_mol.GetNumBonds() == 1
    assert get_bonds_as_tuples(cleaned_ts_mol) == [(2, 3)]


@pytest.mark.parametrize("smi", ["CCO", "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"])
@pytest.mark.parametrize("atom_index", [True, False])
@pytest.mark.parametrize("viewer_size", [(100, 100), (200, 400), (400, 400)])
def test_mol_viewer(smi, atom_index, viewer_size):

    mol = get_mol_with_conformer(smi, 10)

    # Try different conformer id
    n_conf = mol.GetNumConformers()
    for i in range(n_conf):
        viewer = mol_viewer(mol, i, atom_index=atom_index, viewer_size=viewer_size)
        assert isinstance(viewer, py3Dmol.view)

    # Wrong conf id
    with pytest.raises(ValueError):
        mol_viewer(mol, n_conf + 1, atom_index=atom_index, viewer_size=viewer_size)


@pytest.mark.parametrize("atom_index", [True, False])
@pytest.mark.parametrize("viewer_size", [(200, 400), (400, 400)])
@pytest.mark.parametrize("interval", [60, 10000])
@pytest.mark.parametrize("reps", [0, 10])
@pytest.mark.parametrize("step", [1, 5])
@pytest.mark.parametrize("loop", ["forward", "backward", "backAndForth"])
def test_mol_animation(atom_index, viewer_size, interval, reps, step, loop):

    mols = [
        get_mol_with_conformer(smi, 10)
        for smi in ["CO", "CCO", "CCCO", "CCCCO"]
    ]

    mol_animation(
        mols, atom_index=atom_index, viewer_size=viewer_size,
        loop=loop, interval=interval, reps=reps, step=step)


@pytest.mark.parametrize(
    "smi",
    [
        "CCO",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    ],
)
@pytest.mark.parametrize("atom_index", [True, False])
@pytest.mark.parametrize("viewer_size", [(100, 100), (200, 400)])
@pytest.mark.parametrize("opacity", [0.2, 0.8])
def test_conformer_viewer(smi, atom_index, viewer_size, opacity):

    mol = get_mol_with_conformer(smi, 10)

    viewer = conformer_viewer(mol, atom_index=atom_index, viewer_size=viewer_size)
    assert isinstance(viewer, py3Dmol.view)

    # todo: need a better way
    for _ in range(5):
        conf_ids = random.sample(
            list(range(mol.GetNumConformers())),
            random.randint(1, mol.GetNumConformers())
        )
        highlight_ids = [random.choice(conf_ids)]
        viewer = conformer_viewer(mol, conf_ids, highlight_ids, opacity)
        assert isinstance(viewer, py3Dmol.view)


@pytest.mark.parametrize(
    "smi",
    [
        "CCO",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    ],
)
@pytest.mark.parametrize("atom_index", [True, False])
@pytest.mark.parametrize("viewer_size", [(200, 400), (400, 400)])
@pytest.mark.parametrize("interval", [60, 10000])
@pytest.mark.parametrize("reps", [0, 10])
@pytest.mark.parametrize("step", [1, 5])
@pytest.mark.parametrize("loop", ["forward", "backward", "backAndForth"])
def test_conformer_animation(smi, atom_index, viewer_size, interval, reps, step, loop):

    mol = get_mol_with_conformer(smi, 10)

    conf_ids = random.sample(
        list(range(mol.GetNumConformers())), random.randint(1, mol.GetNumConformers())
    )

    viewer = conformer_animation(
        mol, conf_ids, atom_index=atom_index, viewer_size=viewer_size,
        loop=loop, interval=interval, reps=reps, step=step)
    assert isinstance(viewer, py3Dmol.view)


@pytest.mark.parametrize("viewer_grid", [(1, 1), (1, 4), (4, 1), (4, 4)])
@pytest.mark.parametrize("linked", [True, False])
@pytest.mark.parametrize("viewer_size", [None, (200, 400), (400, 400)])
def test_grid_viewer(viewer_grid, linked, viewer_size):

    viewer = grid_viewer(viewer_grid, linked, viewer_size)
    assert viewer.viewergrid is not None


@pytest.mark.parametrize(
    "xyz, dxdydz",
    [
        (
            """2
            [Geometry 1]
            H      0.0000000000    0.0000000000    0.3720870000
            H      0.0000000000    0.0000000000   -0.3720870000""",
            [
                [0.0, 0.0, 0.71],
                [0.0, 0.0, -0.71],
            ],
        ),
    ],
)
@pytest.mark.parametrize("viewer_size", [(200, 400), (400, 400)])
@pytest.mark.parametrize("frames", [20, 100])
@pytest.mark.parametrize("amplitude", [0.1, 10])
@pytest.mark.parametrize("atom_index", [True, False])
def test_freq_viewer(xyz, dxdydz, viewer_size, frames, amplitude, atom_index):

    combined_xyz = merge_xyz_dxdydz(xyz, dxdydz)
    viewer = freq_viewer(
        combined_xyz, frames, amplitude,
        viewer_size=viewer_size, atom_index=atom_index,
    )
    assert isinstance(viewer, py3Dmol.view)


@pytest.mark.parametrize("broken_bonds", [[(1, 2)], []])
@pytest.mark.parametrize("formed_bonds", [[(0, 1)], []])
@pytest.mark.parametrize("color", ["blue", "red"])
@pytest.mark.parametrize("width", [0.05, 0.25])
@pytest.mark.parametrize("viewer_size", [(200, 400), (400, 400)])
@pytest.mark.parametrize("atom_index", [True, False])
def test_ts_viewer(broken_bonds, formed_bonds, color, width, viewer_size, atom_index):

    tsmol = get_mol_with_conformer("[H].[H][C][H]", 10)

    viewer = ts_viewer(
        tsmol, broken_bonds, formed_bonds,
        broken_bond_color=color,
        formed_bond_color=color,
        broken_bond_width=width,
        formed_bond_width=width,
        viewer_size=viewer_size, atom_index=atom_index,
    )

    assert isinstance(viewer, py3Dmol.view)


@pytest.mark.parametrize(
    "rsmi, psmi",
    [("[H].[H][C][H]", "[H][H].[C][H]")]
)
@pytest.mark.parametrize("color", ["blue", "red"])
@pytest.mark.parametrize("width", [0.05, 0.25])
@pytest.mark.parametrize("viewer_size", [(200, 400), (400, 400)])
@pytest.mark.parametrize("atom_index", [True, False])
@pytest.mark.parametrize("linked", [True, False])
@pytest.mark.parametrize("alignment", ["horizontal", "vertical"])
def test_reaction_viewer(rsmi, psmi, color, width, viewer_size, atom_index, linked, alignment):

    rmol = get_mol_with_conformer(rsmi, 10)
    pmol = get_mol_with_conformer(psmi, 10)

    viewer = reaction_viewer(
        rmol, pmol,
        viewer_size=viewer_size,
        atom_index=atom_index,
        linked=linked,
        alignment=alignment,
        formed_bond_color=color,
        broken_bond_color=color,
        formed_bond_width=width,
        broken_bond_width=width,
    )
    assert isinstance(viewer, py3Dmol.view)

    viewer = reaction_viewer(
        rmol,
        pmol,
        rmol,
        viewer_size=viewer_size,
        atom_index=atom_index,
        linked=linked,
        alignment=alignment,
        formed_bond_color=color,
        broken_bond_color=color,
        formed_bond_width=width,
        broken_bond_width=width,
    )
    assert isinstance(viewer, py3Dmol.view)
