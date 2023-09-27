#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2023 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################


from rdmc import RDKitMol
from rdmc.resonance.pathfinder import (
    compute_atom_distance,
    find_adj_lone_pair_multiple_bond_delocalization_paths,
    find_adj_lone_pair_radical_delocalization_paths,
    find_adj_lone_pair_radical_multiple_bond_delocalization_paths,
    find_allyl_delocalization_paths,
    find_allyl_end_with_charge,
    find_butadiene,
    find_butadiene_end_with_charge,
    find_lone_pair_multiple_bond_paths,
    find_N5dc_radical_delocalization_paths,
    # find_shortest_path,
)


class TestFindButadiene:
    def test_13butadiene(self):
        mol = RDKitMol.FromSmiles("C=CC=C")  # 1,3-butadiene

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(3)
        path = find_butadiene(start, end)
        assert path is not None

    def test_acrolein(self):
        mol = RDKitMol.FromSmiles("C=CC=O")  # Acrolein

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(3)
        path = find_butadiene(start, end)
        assert path is not None

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(4)  # wrong end
        path = find_butadiene(start, end)
        assert path is None

    def test_135hexatriene(self):
        mol = RDKitMol.FromSmiles("C=CC=CC=C")  # 1,3,5-hexatriene

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(5)
        path = find_butadiene(start, end)
        assert path is not None

    def test_13cyclohexadiene(self):
        mol = RDKitMol.FromSmiles('C1=CC=CCC1')  # 1,3-cyclohexadiene

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(3)
        path = find_butadiene(start, end)
        assert path is not None

    def test_14cyclohexadiene(self):
        mol = RDKitMol.FromSmiles('C1=CCC=CC1')  # 1,4-cyclohexadiene

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(3)
        path = find_butadiene(start, end)
        assert path is None

    def test_benzene(self):
        mol = RDKitMol.FromSmiles("C1=CC=CC=C1")  # benzene

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(5)
        path = find_butadiene(start, end)
        assert path is not None

    def test_c4h4(self):
        mol = RDKitMol.FromSmiles("C=C=C=C")  # C4H4

        start, end = mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(3)
        path = find_butadiene(start, end)
        assert path is not None


class TestFindAllylEndWithCharge:
    def test_c2h2o3(self):
        mol = RDKitMol.FromSmiles('[C:1](=[O+:5][C:2](=[O:3])[O-:4])([H:6])[H:7]')
        start = mol.GetAtomWithIdx(2)
        paths = find_allyl_end_with_charge(start)
        idx_path = sorted([[atom.GetIdx() + 1 for atom in path[0::2]] for path in paths])

        expected_idx_path = [[3, 2, 4], [3, 2, 5]]
        assert idx_path == expected_idx_path

    def test_c3h2(self):
        inchi = "InChI=1S/C3H2/c1-3-2/h1-2H"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(0)
        path = find_allyl_end_with_charge(start)[0]
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [1, 3, 2]
        assert idx_path == expected_idx_path

    def test_c3h4(self):
        inchi = "InChI=1S/C3H4/c1-3-2/h1,3H,2H2"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(0)
        path = find_allyl_end_with_charge(start)[0]
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [1, 3, 2]
        assert idx_path == expected_idx_path

    def test_c3h2o3(self):
        mol = RDKitMol.FromSmiles('[C:1](=[C:2]=[C:3]([O-:4])[O+:6]=[O:5])([H:7])[H:8]')
        start = mol.GetAtomWithIdx(1)
        paths = find_allyl_end_with_charge(start)
        idx_paths = sorted([[atom.GetIdx() + 1 for atom in path[0::2]] for path in paths])
        idx_paths = sorted(idx_paths)

        expected_idx_paths = [[2, 3, 4], [2, 3, 6]]
        assert idx_paths == expected_idx_paths

    def test_c3h4o4(self):
        inchi = "InChI=1S/C3H4O4/c4-3(5)1-2-7-6/h1-3,6H"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(6)
        path = find_allyl_end_with_charge(start)[0]
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [7, 2, 1]
        assert idx_path == expected_idx_path

    def test_c5h6o(self):
        inchi = "InChI=1S/C5H6O/c6-5-3-1-2-4-5/h1-3,5H,4H2"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(1)
        path = find_allyl_end_with_charge(start)[0]
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [2, 1, 3]
        assert idx_path == expected_idx_path


class TestFindButadieneEndWithCharge:
    def test_co(self):
        mol = RDKitMol.FromSmiles('[C-]#[O+]')
        start = mol.GetAtomWithIdx(0)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [1, 2]
        assert idx_path == expected_idx_path

    def test_c2h2o3(self):
        mol = RDKitMol.FromSmiles('[C:1](=[O+:5][C:2](=[O:3])[O-:4])([H:6])[H:7]')
        start = mol.GetAtomWithIdx(0)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [1, 5]
        assert idx_path == expected_idx_path

    def test_c3h2o3(self):
        mol = RDKitMol.FromSmiles('[C:1](=[C:2]=[C:3]([O-:4])[O+:6]=[O:5])([H:7])[H:8]')
        start = mol.GetAtomWithIdx(4)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [5, 6]
        assert idx_path == expected_idx_path

    def test_c4h6o(self):
        mol = RDKitMol.FromSmiles('[C:1]([C-:2]([C:3]([C:4]#[O+:5])([H:10])[H:11])[H:9])([H:6])([H:7])[H:8]')
        start = mol.GetAtomWithIdx(3)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [4, 5]
        assert idx_path == expected_idx_path

    def test_c5h6o2(self):
        mol = RDKitMol.FromSmiles('[C-:1]([C:5]1([H:12])[C:3]([H:10])=[C:2]([H:9])[C:4]([H:11])=[O+:6]1)([H:7])[H:8]')
        start = mol.GetAtomWithIdx(2)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [3, 2, 4, 6]
        assert idx_path == expected_idx_path

    def test_c8h14o4(self):
        inchi = "InChI=1S/C8H14O4S/c1-3-6-13(2,11)7-8(9)4-5-12-10/h3,6H,1,4-5,7H2,2H3,(H-,10,11)"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(0)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]
        expected_idx_path = [1, 3, 6, 13]
        assert idx_path == expected_idx_path

    def test_c6h6o6(self):
        inchi = "InChI=1S/C6H6O6/c7-6(2-5-12-9)10-3-1-4-11-8/h1,7H,4-5H2"
        mol = RDKitMol.FromInchi(inchi)
        start = mol.GetAtomWithIdx(2)
        path = find_butadiene_end_with_charge(start)
        idx_path = [atom.GetIdx() + 1 for atom in path[0::2]]

        expected_idx_path = [3, 10]
        assert idx_path == expected_idx_path


class TestDistanceComputing:
    def test_2_atoms(self):
        smi = "CCC"
        mol = RDKitMol.FromSmiles(smi)
        atom_indices = [1, 2]
        distances = compute_atom_distance(atom_indices, mol)

        expected = {(1, 2): 1}
        assert distances == expected

    def test_3_atoms(self):
        smi = "CCC"
        mol = RDKitMol.FromSmiles(smi)
        atom_indices = [1, 2, 3]
        distances = compute_atom_distance(atom_indices, mol)

        expected = {
            (1, 2): 1,
            (1, 3): 2,
            (2, 3): 1,
        }
        assert distances == expected


class TestFindAllylDelocalizationPaths:
    """
    test the find_allyl_delocalization_paths method
    """

    def test_allyl_radical(self):
        smiles = "[CH2]C=C"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_allyl_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_nitrogenated_birad(self):
        smiles = "[N]C=[CH]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_allyl_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths


class TestFindLonePairMultipleBondPaths:
    """
    test the find_lone_pair_multiple_bond_paths method
    """

    def test_azide(self):
        smiles = "[N-]=[N+]=N"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(2))
        assert paths

    def test_nh2cho(self):
        smiles = "NC=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_n2oa(self):
        smiles = "[N-]=[N+]=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_n2ob(self):
        smiles = "N#[N+][O-]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(2))
        assert paths

    def test_hn3(self):
        smiles = "[NH-][N+]#N"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_sn2(self):
        smiles = "OS(O)=[N+]=[N-]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(2))
        assert paths

    def test_h2nnoo(self):
        smiles = "N[N+]([O-])=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.GetAtomWithIdx(0))
        assert paths


class TestFindAdjLonePairRadicalDelocalizationPaths:
    """
    test the find_lone_pair_radical_delocalization_paths method
    """

    def test_no2a(self):
        smiles = "[O]N=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_no2b(self):
        smiles = "[O-][N+]=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.GetAtomWithIdx(1))
        assert paths

    def test_hoso(self):
        smiles = "[O]SO"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths

    def test_double_bond(self):
        mol = RDKitMol.FromSmiles('[O+:1]=[N-:2]')
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths


class TestFindAdjLonePairMultipleBondDelocalizationPaths:
    """
    test the find_lone_pair_multiple_bond_delocalization_paths method
    """

    def test_sho3(self):
        smiles = "O=[SH](=O)[O]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_multiple_bond_delocalization_paths(mol.GetAtomWithIdx(0))
        assert paths


class TestFindAdjLonePairRadicalMultipleBondDelocalizationPaths:
    """
    test the find_lone_pair_radical_multiple_bond_delocalization_paths method
    """

    def test_ns(self):
        smiles = "N#[S]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_multiple_bond_delocalization_paths(mol.GetAtomWithIdx(1))
        assert paths

    def test_hso3(self):
        smiles = "O[S](=O)=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_multiple_bond_delocalization_paths(mol.GetAtomWithIdx(1))
        assert paths


class TestFindN5dcRadicalDelocalizationPaths:
    """
    test the find_N5dc_radical_delocalization_paths method
    """

    def test_hnnoo(self):
        smiles = "N=[N+]([O])([O-])"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_N5dc_radical_delocalization_paths(mol.GetAtomWithIdx(1))
        assert paths
