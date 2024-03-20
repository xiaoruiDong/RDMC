from rdmc.mol.helper import parse_xyz_or_smiles_list

def test_parse_xyz_or_smiles_list():
    """
    Test the function that parses a list of xyz or smiles strings.
    """
    mol_list = [
        {'smi': "CCC"},
        {'xyz': "H 0 0 0"},
        {'smi': "[CH2]", 'mult': 1},
    ]
    mols = parse_xyz_or_smiles_list(
        mol_list,
        header=False,
        backend="xyz2mol",
    )

    assert len(mols) == 3
    assert mols[0]["mol"].ToSmiles() == "CCC"
    assert mols[1]["mol"].ToSmiles() == "[H]"
    assert mols[2]["mol"].ToSmiles() == "[CH2]"
    assert mols[2]["mol"].GetSpinMultiplicity() == 1
    assert [mol["is3D"] for mol in mols] == [False, True, False]
