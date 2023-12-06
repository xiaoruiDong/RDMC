import pytest

from rdkit import Chem

from rdmc.rdtools.mol import get_spin_multiplicity
from rdmc.rdtools.fix import saturate_mol


smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


@pytest.mark.parametrize(
    "smi, exp_smi, exp_mult",
    [
        ("[CH2][CH2]", "C=C", 1),
        ("[CH2][CH]CC[CH][CH2]", "C=CCCC=C", 1),
        ("[CH2]", "[CH2]", 1),
        ("[NH]", "[NH]", 1),
        ("[CH]CC[CH]", "[CH]CC[CH]", 1),
        ("[CH2]C=C[CH2]", "C=CC=C", 1),
        ("[CH]C=CC=C[CH2]", "[CH]=CC=CC=C", 2),
    ],
)
def test_saturate_mol(smi, exp_smi, exp_mult):
    mol = Chem.MolFromSmiles(smi, smi_params)

    saturate_mol(mol)

    assert get_spin_multiplicity(mol) == exp_mult
    assert Chem.MolToSmiles(mol) == exp_smi
    # assert saturate_mol(mol, spin_multiplicity).ToSmiles() == exp_smi
