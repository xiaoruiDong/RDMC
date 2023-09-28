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


from rdmc.resonance.filtration import (
    get_octet_deviation_list,
    get_octet_deviation,
    filter_structures,
    charge_filtration,
    get_charge_span_list,
    aromaticity_filtration,
)

from rdmc.resonance.utils import get_charge_span

from rdmc import RDKitMol
# from rmgpy.molecule.resonance import generate_resonance_structures, analyze_molecule


class TestFiltration:
    def test_basic_filtration(self):
        """Test that structures with higher octet deviation get filtered out"""
        mol1 = RDKitMol.FromSmiles('[N:1](=[O:2])[O:3]')
        mol2 = RDKitMol.FromSmiles('[N:1]([O+:2])[O-:3]')
        mol3 = RDKitMol.FromSmiles('[O:1][N+:3][O-:2]')

        # to meet the multiplicity defined by RMG
        mol2.GetAtomWithIdx(1).SetNumRadicalElectrons(0)
        mol3.GetAtomWithIdx(2).SetNumRadicalElectrons(0)

        mol_list = [mol1, mol2, mol3]
        octet_deviation_list = get_octet_deviation_list(mol_list)
        filtered_list = filter_structures(mol_list)

        assert octet_deviation_list == [1, 3, 3]
        assert len(filtered_list) == 1
        assert all([atom.GetFormalCharge() == 0 for atom in filtered_list[0].GetAtoms()])

    def test_penalty_for_o4tc(self):
        """Test that an O4tc atomtype with octet 8 gets penalized in the electronegativity heuristic"""
        mol = RDKitMol.FromSmiles('[S:1]([O-:2])#[O+:3]')
        octet_deviation = get_octet_deviation(mol)
        assert octet_deviation == 0
        assert mol.GetBondBetweenAtoms(0, 2).GetBondTypeAsDouble() == 3
        # mol_list = generate_resonance_structures(mol)
        # assert len(mol_list) == 2
        # for mol in mol_list:
        #     if mol.reactive:
        #         for atom in mol.vertices:
        #             assert atom.charge == 0

    def test_penalty_birads_replacing_lone_pairs(self):
        """Test that birads on `S u2 p0` are penalized"""
        adj = """
        multiplicity 3
        1 S u2 p0 c0 {2,D} {3,D}
        2 O u0 p2 c0 {1,D}
        3 O u0 p2 c0 {1,D}
        """
        mol = RDKitMol.FromSmiles('[S:1](=[O:2])=[O:3]')
        mol.GetAtomWithIdx(0).SetNumRadicalElectrons(2)

        # mol_list = generate_resonance_structures(mol, keep_isomorphic=False, filter_structures=True)
        # for mol in mol_list:
        #     if mol.reactive:
        #         for atom in mol.vertices:
        #             if atom.is_sulfur():
        #                 assert atom.radical_electrons != 2
        # assert len(mol_list) == 3
        # assert sum([1 for mol in mol_list if mol.reactive]) == 2

    def test_penalty_for_s_triple_s(self):
        """Test that an S#S substructure in a molecule gets penalized in the octet deviation score"""
        mol = RDKitMol.FromSmiles('[C:1]([S:3](#[S:4][C:2]([H:8])([H:9])[H:10])=[O:11])([H:5])([H:6])[H:7]')
        octet_deviation = get_octet_deviation(mol)
        assert octet_deviation == 1.0

    def test_radical_site(self):
        """Test that a charged molecule isn't filtered if it introduces new radical site"""
        smi1 = '[O:1][N:3]=[O:2]'
        smi2 = '[O-:1][N+:3]=[O:2]'
        smi3 = '[O:1][N+:3][O-:2]'

        mol_list = [
            RDKitMol.FromSmiles(smi1),
            RDKitMol.FromSmiles(smi2),
            RDKitMol.FromSmiles(smi3),
        ]
        mol_list[2].GetAtomWithIdx(2).SetNumRadicalElectrons(0)

        filtered_list = charge_filtration(mol_list, get_charge_span_list(mol_list))
        assert len(filtered_list) == 2
        assert any([get_charge_span(mol) == 1 for mol in filtered_list])
        for mol in filtered_list:
            if get_charge_span(mol) == 1:
                for atom in mol.GetAtoms():
                    if atom.GetFormalCharge() == -1:
                        assert atom.GetAtomicNum() == 8
                    if atom.GetFormalCharge() == 1:
                        assert atom.GetAtomicNum() == 7

    def test_electronegativity(self):
        """Test that structures with charge separation are only kept if they obey the electronegativity rule

        (If a structure must have charge separation, negative charges will be assigned to more electronegative atoms,
        whereas positive charges will be assigned to less electronegative atoms)

        In this test, only the three structures with no charge separation and the structure where both partial charges
        are on the nitrogen atoms should be kept."""
        smi1 = '[N:1]([N:2][H:4])=[S:3]=[O:5]'
        smi2 = '[N:1](=[N:2][H:4])[S:3]=[O:5]'
        smi3 = '[N:1](=[N:2][H:4])[S:3][O:5]'
        smi4 = '[N+:1]([N-:2][H:4])=[S:3]=[O:5]'
        smi5 = '[N+:1](=[N:2][H:4])[S-:3]=[O:5]'
        smi6 = '[N:1]([N-:2][H:4])[S+:3]=[O:5]'
        smi7 = '[N+:1](=[N:2][H:4])[S:3][O-:5]'

        mol_list = [
            RDKitMol.FromSmiles(smi1),
            RDKitMol.FromSmiles(smi2),
            RDKitMol.FromSmiles(smi3),
            RDKitMol.FromSmiles(smi4),
            RDKitMol.FromSmiles(smi5),
            RDKitMol.FromSmiles(smi6),
            RDKitMol.FromSmiles(smi7),
        ]

        filtered_list = charge_filtration(mol_list, get_charge_span_list(mol_list))
        assert len(filtered_list) == 4
        assert any([get_charge_span(mol) == 1 for mol in filtered_list])
        for mol in filtered_list:
            if get_charge_span(mol) == 1:
                for atom in mol.GetAtoms():
                    if abs(atom.GetFormalCharge()) == 1:
                        assert atom.GetAtomicNum() == 7

    def test_aromaticity(self):
        """Test that aromatics are properly filtered."""
        smi1 = '[C:1]1([C:7]([H:13])[H:14])=[C:2]([H:8])[C:4]([H:9])=[C:5]([H:10])[C:6]([H:11])=[C:3]1[H:12]'
        smi2 = '[c:1]1([C:7]([H:13])[H:14])[c:2]([H:8])[c:4]([H:9])[c:5]([H:10])[c:6]([H:11])[c:3]1[H:12]'
        smi3 = '[C:1]1(=[C:7]([H:13])[H:14])[C:2]([H:8])[C:4]([H:9])=[C:5]([H:10])[C:6]([H:11])=[C:3]1[H:12]'
        smi4 = '[C:1]1(=[C:7]([H:13])[H:14])[C:2]([H:8])=[C:5]([H:9])[C:4]([H:10])[C:6]([H:11])=[C:3]1[H:12]'

        mol_list = [
            RDKitMol.FromSmiles(smi1),
            RDKitMol.FromSmiles(smi2),
            RDKitMol.FromSmiles(smi3),
            RDKitMol.FromSmiles(smi4),
        ]

        # filtered_list = aromaticity_filtration(mol_list, analyze_molecule(mol_list[0]))
        # assert len(filtered_list) == 3
