import pytest
import numpy as np

from rdmc import RDKitMol


def test_add_null_conformer():
    """
    Test adding null conformers to a molecule.
    """
    smi = '[C:2]([H:3])([H:4])([H:5])[H:6].[H:1]'
    mol = RDKitMol.FromSmiles(smi)
    assert mol.GetNumConformers() == 0

    # Test adding 1 null conformer with default arguments
    mol.AddNullConformer()
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer(id=0)
    coords = conf.GetPositions()
    assert coords.shape == (6, 3)
    # Test all atom coordinates are non-zero
    assert np.all(np.all(coords != np.zeros((6, 3)), axis=1))

    # Test adding a conformer with a specific id and all zero
    mol.AddNullConformer(confId=10, random=False)
    assert mol.GetNumConformers() == 2
    with pytest.raises(ValueError):
        mol.GetConformer(id=1)
    conf = mol.GetConformer(id=10)
    coords = conf.GetPositions()
    assert coords.shape == (6, 3)
    assert np.all(np.equal(coords, np.zeros((6, 3))))