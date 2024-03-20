import pytest

from rdmc import RDKitMol
from rdmc.conformer_generation.embedders import ETKDGEmbedder


@pytest.fixture
def etkdg_embedder():
    return ETKDGEmbedder()


@pytest.mark.parametrize(
    "smi",
    [
        "[C:1]([C@@:2]([O:3][H:12])([C:4]([N:5]([C:6](=[O:7])[H:16])"
        "[H:15])([H:13])[H:14])[H:11])([H:8])([H:9])[H:10]",
        "CN1C2=C(C=C(C=C2)Cl)C(=NCC1=O)C3=CC=CC=C3",
    ]
)
@pytest.mark.parametrize(
    "n_conf",
    [1, 5, 10, 20],
)
def test_embedding_etkdg(etkdg_embedder, smi, n_conf):

    assert len(etkdg_embedder(smi, n_conf)) == n_conf
