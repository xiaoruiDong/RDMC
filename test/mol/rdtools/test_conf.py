import pytest

import numpy as np

from rdmc.rdtools.conf import (
    add_null_conformer,
    create_conformer,
    embed_multiple_null_confs,
    reflect,
    set_conformer_coordinates,
)

from rdkit import Chem
from rdmc.mol import RDKitMol


@pytest.mark.parametrize(
    "coords, conf_id",
    [
        (np.array([[0, 0, 0]], dtype=float), 0),
        (np.array([[0, 0, 0], [1, 1, 1]], dtype=float), 1),
        (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float), 2),
    ],
)
def test_create_conformer(coords, conf_id):
    conf = create_conformer(coords, conf_id)

    assert conf.GetNumAtoms() == coords.shape[0]
    assert conf.GetId() == conf_id
    np.testing.assert_array_equal(conf.GetPositions(), coords)


@pytest.mark.parametrize(
    "smi, conf_id",
    [
        ("CC", 1),
        ("c1ccccc1", 2),
        ("CCO", 3),
    ],
)
def test_add_null_conformer(smi, conf_id):
    mol = RDKitMol.FromSmiles(smi)
    add_null_conformer(mol, conf_id=conf_id, random=False)

    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().GetId() == conf_id
    np.testing.assert_array_equal(
        mol.GetConformer().GetPositions(), np.zeros((mol.GetNumAtoms(), 3))
    )


def test_reflect():
    for i in range(10):
        coords = np.random.rand(i, 3)
        conf = Chem.Conformer(i)
        set_conformer_coordinates(conf, coords)
        reflect(conf)
        reflected_coords = conf.GetPositions()
        np.testing.assert_array_equal(reflected_coords, coords * np.array([-1, 1, 1]))


def test_embed_multiple_null_confs():
    n_conformers = 10
    mol = RDKitMol.FromSmiles("CC")
    embed_multiple_null_confs(mol, n=n_conformers, random=True)
    assert mol.GetNumConformers() == n_conformers

    # Expect conformer are added with continuous IDs
    assert [mol.GetConformer(i) for i in range(10)]
