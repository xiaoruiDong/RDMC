from rdmc import RDKitMol
from rdtools.resonance.utils import get_lone_pair, is_aromatic, is_identical
from rdtools.resonance.rmg_backend import (
    _clar_optimization,
    _clar_transformation,
    generate_clar_structures,
    generate_kekule_structure,
    generate_optimal_aromatic_resonance_structures,
    generate_resonance_structures,
)
import rdkit
from rdkit import Chem

import pytest

smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


class TestResonance:
    def test_allyl_shift(self):
        """Test allyl shift for hexadienyl radical"""
        mol = Chem.MolFromSmiles("C=C[CH]C=CC", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_trirad_allyl_shift(self):
        """Test allyl shift for a tri-rad carbon"""
        mol = Chem.MolFromSmiles("[C]N=N", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3
        assert any(
            [
                any([atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )  # expecting [C]=[N+.][NH-]

    def test_oxime(self):
        """Test resonance structure generation for CC=N[O] radical

        Simple case for lone pair <=> radical resonance"""
        mol = Chem.MolFromSmiles("CC=N[O]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3
        assert any(
            [
                any([atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )

    def test_ring_allyl_shift(self):
        """Test allyl shift for a cyclic species with heteroatoms"""
        mol = Chem.MolFromSmiles("[CH]1C=NC=N1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 5

    def test_carbene_allyl_shift(self):
        """Test allyl shift for a carbene species"""
        mol = Chem.MolFromSmiles("[C]=C=O", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_ch2chcho(self):
        """Test resonance structure generation for C=C[CH][O] bi-radical"""
        mol = Chem.MolFromSmiles("C=C[CH][O]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_ch2no(self):
        """Test combined resonance transitions of allyl-shift and lonePair-radical"""
        mol = Chem.MolFromSmiles("[CH2]N=O", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_ch3s2o2(self):
        """Test combined resonance transitions of one_pair_radical_multiple_bond"""
        mol = Chem.MolFromSmiles("CSS(=O)[O]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_n2so2(self):
        """Test the resonance transitions of a species with several hereroatoms and several multiple bonds"""
        mol = Chem.MolFromSmiles("[N-]=[N+]=S(=O)=O", smi_params)
        mol_list = generate_resonance_structures(mol)
        # Note: the reference value in the original RMG test is 2
        # However, given Xiaorui's test on RMG-Py and RMG-database,
        # the reference value should be 1
        assert len(mol_list) == 1

    def test_nsh(self):
        """Test that a resonance structure with a minimal octet deviation but higher charge span is filtered out"""
        mol = Chem.MolFromSmiles("N#S", smi_params)
        # mol = Chem.MolFromSmiles("N#S")._mol  # RDMC force no implicit
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 1
        assert all([atom.GetFormalCharge() == 0 for atom in mol_list[0].GetAtoms()])

    def test_nco(self):
        """Test resonance structure generation for NCO

        NCO should only have two resonance structures [N.]=C=O <=> N#C[O.], and not a third structure which has
        the same octet deviation, has a charge separation, but no ne radical site: [N+.]#C[O-]
        """
        mol = Chem.MolFromSmiles("[N]=C=O", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2
        assert all(
            [
                all([atom.GetFormalCharge() == 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )  # none of the
        # structures should be charged

    def test_no2(self):
        """Test resonance structure generation for [O]N=O radical

        Test case for the lone pair <=> radical resonance transition.
        Also tests that the filtering function allows charge separation when the radical site is changed.
        """
        mol = Chem.MolFromSmiles("[O]N=O", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2
        assert any(
            [
                any([atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )  # one of the
        # structures should be charged

    def test_n2o(self):
        """Test resonance structure generation for N#[N+][O-]

        A classic N5ddc <=> N5tc resonance transition"""
        mol = Chem.MolFromSmiles("N#[N+][O-]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2
        assert all(
            [
                any([atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )  # both structures
        # should have some charged atoms

        sbonds = 0
        dbonds = 0
        tbonds = 0
        for mol in mol_list:
            for atom in mol.GetAtoms():
                for bond in atom.GetBonds():
                    if bond.GetBondType() == 1:
                        sbonds += 1
                    elif bond.GetBondType() == 2:
                        dbonds += 1
                    elif bond.GetBondType() == 3:
                        tbonds += 1
        assert sbonds / 2 == 1  # each bond is counted twice above
        assert dbonds / 2 == 2
        assert tbonds / 2 == 1

    def test_azide(self):
        """Test resonance structure generation for ethyl azide

        Simple case for N5ddc <=> N5tc resonance
        Azides are described by three resonance structures: N=[N+]=[N-] <=> [NH-][N+]#N <=> [NH+]#[N+][N-2]
        """
        mol = Chem.MolFromSmiles("CCN=[N+]=[N-]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2
        assert all(
            [
                any([atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()])
                for mol in mol_list
            ]
        )

    def test_ozone(self):
        """Test resonance structure generation for O3, S3 and SO2.

        Compare that these iso-electronic structures have the same number of resonance structures
        """
        mol = Chem.MolFromSmiles("[O-][O+]=O", smi_params)
        mol_list_1 = generate_resonance_structures(mol)
        assert len(mol_list_1) == 1
        mol = Chem.MolFromSmiles("O=S=O", smi_params)
        mol_list_2 = generate_resonance_structures(mol)
        assert len(mol_list_2) == 1
        mol = Chem.MolFromSmiles("S=S=S", smi_params)
        mol_list_3 = generate_resonance_structures(mol)
        assert len(mol_list_3) == 1

    def test_hco_vs_hcs(self):
        """Test resonance structure generation for [CH]=O and [CH]=S

        These iso-electronic structures have a different(!) number of resonance structures
        """
        mol = Chem.MolFromSmiles("[CH]=O", smi_params)
        mol_list_1 = generate_resonance_structures(mol)
        assert len(mol_list_1) == 1
        mol = Chem.MolFromSmiles("[CH]=S", smi_params)
        mol_list_2 = generate_resonance_structures(mol)
        assert len(mol_list_2) == 2

    def test_no(self):
        """Test that an incorrect NO structure [::N][::O.] is correctly identified as [:N.]=[::O]

        The incorrect structure could be generated from HON (O[::N]) during an RMG run, and should be identified as NO.
        The original structure should be kept as unreactive (appended at the end of the molecule list)
        """
        mol = Chem.MolFromSmiles("[N][O]", smi_params)
        mol.GetAtomWithIdx(0).SetNumRadicalElectrons(0)
        mol_list = generate_resonance_structures(mol)

        assert len(mol_list) == 2
        # RDMC has a different ordering approach, input always first
        # So the lone pair reference values are swapped
        assert (
            get_lone_pair(mol_list[0].GetAtoms()[0])
            + get_lone_pair(mol_list[0].GetAtoms()[1])
            == 4
        )
        assert (
            get_lone_pair(mol_list[1].GetAtoms()[0])
            + get_lone_pair(mol_list[1].GetAtoms()[1])
            == 3
        )

    def test_n5dc_radical(self):
        """Test the N5dc radical resonance transformation

        We should see N=[N+]([O])([O-]) <=> [NH-][N+]([O])=O
        Two isomorphic structures should be included in mol_list: N=[N+]([O])([O-]) <=> N=[N+]([O-])([O])
        """
        mol = Chem.MolFromSmiles("N=[N+]([O-])[O]", smi_params)
        mol_list = generate_resonance_structures(mol, keep_isomorphic=True)
        assert len(mol_list) == 6
        isomorphic_counter = 0
        negatively_charged_nitrogen = 0
        for mol1 in mol_list:
            if mol1.GetSubstructMatch(mol):
                isomorphic_counter += 1
            for atom in mol1.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() < 0:
                    negatively_charged_nitrogen += 1
        assert isomorphic_counter == 2
        assert negatively_charged_nitrogen == 2

    def test_n5dc(self):
        """Test the N5dc resonance transformation

        We should see N[N+]([O-])=O <=> N[N+](=O)[O-], which are isomorphic"""
        mol = Chem.MolFromSmiles("N[N+]([O-])=O", smi_params)
        mol_list = generate_resonance_structures(mol, keep_isomorphic=True)
        assert len(mol_list) == 2
        assert mol_list[0].GetSubstructMatch(mol_list[1])

    def test_styryl1(self):
        """Test resonance structure generation for styryl, with radical on branch

        In this case, the radical can be delocalized into the aromatic ring"""
        mol = Chem.MolFromSmiles("c1ccccc1[C]=C", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 4

    def test_styryl2(self):
        """Test resonance structure generation for styryl, with radical on ring

        In this case, the radical can be delocalized into the aromatic ring"""
        mol = Chem.MolFromSmiles("C=C=C1C=C[CH]C=C1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_naphthyl(self):
        """Test resonance structure generation for naphthyl radical

        In this case, the radical is orthogonal to the pi-orbital plane and cannot delocalize
        """
        mol = Chem.MolFromSmiles("c12[c]cccc1cccc2", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 4

    def test_methyl_napthalene(self):
        """Test resonance structure generation for methyl naphthalene

        Example of stable polycyclic aromatic species"""
        mol = Chem.MolFromSmiles("CC1=CC=CC2=CC=CC=C12", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 4

    def test_methyl_phenanthrene(self):
        """Test resonance structure generation for methyl phenanthrene

        Example of stable polycyclic aromatic species"""
        mol = Chem.MolFromSmiles("CC1=CC=CC2C3=CC=CC=C3C=CC=21", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 3

    def test_methyl_phenanthrene_radical(self):
        """Test resonance structure generation for methyl phenanthrene radical

        Example radical polycyclic aromatic species where the radical can delocalize"""
        mol = Chem.MolFromSmiles("[CH2]C1=CC=CC2C3=CC=CC=C3C=CC=21", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 9

    def test_aromatic_with_lone_pair_resonance(self):
        """Test resonance structure generation for aromatic species with lone pair <=> radical resonance"""
        mol = Chem.MolFromSmiles("c1ccccc1CC=N[O]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 4

    def test_aromatic_with_n_resonance(self):
        """Test resonance structure generation for aromatic species with lone pair resonance"""
        mol = Chem.MolFromSmiles("c1ccccc1CCN=[N+]=[N-]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_aromatic_with_o_n_resonance(self):
        """Test resonance structure generation for aromatic species with heteroatoms

        This test was specifically designed to recreate RMG-Py issue #1598.
        Key conditions: having heteroatoms, starting with aromatic structure, and keep_isomorphic=True
        """
        mol = Chem.MolFromSmiles("[O-][N+](c1ccccc1)(O)[O]", smi_params)
        mol_list = generate_resonance_structures(mol, keep_isomorphic=True)
        # Xiaorui: I think RMG behaves incorrectly here (`1` in the RMG test)
        assert len(mol_list) == 2

    def test_no_clar_structures(self):
        """Test that we can turn off Clar structure generation."""
        mol = Chem.MolFromSmiles("C1=CC=CC2C3=CC=CC=C3C=CC=21", smi_params)
        mol_list = generate_resonance_structures(mol, clar_structures=False)
        assert len(mol_list) == 2

    def test_c13h11_rad(self):
        """Test resonance structure generation for p-methylbenzylbenzene radical

        Has multiple resonance structures that break aromaticity of a ring"""
        mol = Chem.MolFromSmiles("[CH](c1ccccc1)c1ccc(C)cc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        # RMG has the one with two kekule benzene and one with two aromatic benzene
        # RDMC see them as the same therefore has one less resonance structure
        assert len(mol_list) == 6

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_c8h8(self):
        """Test resonance structure generation for 5,6-dimethylene-1,3-cyclohexadiene

        Example of molecule that RDKit considers aromatic, but RMG does not"""
        mol = Chem.MolFromSmiles("C=C1C=CC=CC1=C", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 1

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_c8h7_j(self):
        """Test resonance structure generation for 5,6-dimethylene-1,3-cyclohexadiene radical

        Example of molecule that RDKit considers aromatic, but RMG does not"""
        mol = Chem.MolFromSmiles("C=C1C=CC=CC1=[CH]", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 1

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_c8h7_j2(self):
        """Test resonance structure generation for 5,6-dimethylene-1,3-cyclohexadiene radical

        Example of molecule that RDKit considers aromatic, but RMG does not"""
        mol = Chem.MolFromSmiles("C=C1C=[C]C=CC1=C", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 1

    def test_c9h9_aro(self):
        """Test cyclopropyl benzene radical, aromatic SMILES"""
        mol = Chem.MolFromSmiles("[CH]1CC1c1ccccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c9h9_kek(self):
        """Test cyclopropyl benzene radical, kekulized SMILES"""
        mol = Chem.MolFromSmiles("[CH]1CC1C1C=CC=CC=1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_benzene_aro(self):
        """Test benzene, aromatic SMILES"""
        mol = Chem.MolFromSmiles("c1ccccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_benzene_kek(self):
        """Test benzene, kekulized SMILES"""
        mol = Chem.MolFromSmiles("C1C=CC=CC=1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c9h11_aro(self):
        """Test propylbenzene radical, aromatic SMILES"""
        mol = Chem.MolFromSmiles("[CH2]CCc1ccccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c10h11_aro(self):
        """Test cyclobutylbenzene radical, aromatic SMILES"""
        mol = Chem.MolFromSmiles("[CH]1CCC1c1ccccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c9h10_aro(self):
        """Test cyclopropylbenzene, aromatic SMILES"""
        mol = Chem.MolFromSmiles("C1CC1c1ccccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c10h12_aro(self):
        """Test cyclopropylmethyl benzene, aromatic SMILES"""
        mol = Chem.MolFromSmiles("C1CC1c1c(C)cccc1", smi_params)
        mol_list = generate_resonance_structures(mol)
        assert len(mol_list) == 2

    def test_c9h10_aro_2(self):
        """Test cyclopropyl benzene, generate aromatic resonance isomers"""
        mol = Chem.MolFromSmiles("C1CC1c1ccccc1", smi_params)
        mol_list = generate_optimal_aromatic_resonance_structures(mol)
        assert len(mol_list) == 1

    def test_aryne_1_ring(self):
        """Test aryne resonance for benzyne"""
        mol1 = Chem.MolFromSmiles("C1=CC=C=C=C1", smi_params)
        mol2 = Chem.MolFromSmiles("C1C#CC=CC=1", smi_params)

        mol_list1 = generate_resonance_structures(mol1)
        assert len(mol_list1) == 2

        # We increase this number by 1 from RMG reference because RDKit considers
        # benzyne (STSDSD) as aromatic, so it will include both arom and kek form
        # of benzyne in the resonance structure list
        mol_list2 = generate_resonance_structures(mol2)
        assert len(mol_list2) == 3

        assert mol_list1[1].GetSubstructMatch(mol2)
        assert mol_list2[2].GetSubstructMatch(mol1)

    def test_aryne_2_rings(self):
        """Test aryne resonance in naphthyne"""
        mol1 = Chem.MolFromSmiles("C12=CC=C=C=C1C=CC=C2", smi_params)
        mol2 = Chem.MolFromSmiles("C12C#CC=CC=1C=CC=C2", smi_params)

        # We increase this number by 1 from RMG reference
        # the difference is due to RDMC thinks mol1 is partially
        # aromatic and therefore keeps the kekulized form of mol1
        mol_list1 = generate_resonance_structures(mol1)
        assert len(mol_list1) == 3

        mol_list2 = generate_resonance_structures(mol2)
        assert len(mol_list2) == 3

        # TODO: So far, the generated structures are not good enough
        # TODO: E.g., may result in atoms of total bond order = 5
        # TODO: Improve the generation and matching for aryne
        # Check that they both have an aromatic resonance form
        assert mol_list1[1].GetSubstructMatch(mol_list2[1])

    def test_aryne_3_rings(self):
        """Test aryne resonance in phenanthryne"""
        mol = Chem.MolFromSmiles("C12C#CC=CC=1C=CC3=C2C=CC=C3", smi_params)

        mol_list = generate_resonance_structures(mol)
        # RMG has 5 resonance structures. In new versions of RDKit like 2023.09, 5 is successfully
        # reproduced, and the structures are consistent. However, in earlier versions like 2021.09,
        # 4 is returned. This is due to different default aromaticity criteria. The differences are not
        # significant enough to implement any kind of remedy to the earlier versions, so we will pass both cases.
        # Given RDKit's version style, it is okay to directly compare the versions as strings.
        if rdkit.__version__ >= '2023.09.1':
            assert len(mol_list) == 5
        else:
            assert len(mol_list) == 4 or len(mol_list) == 5

    def test_fused_aromatic1(self):
        """Test we can make aromatic perylene from both adjlist and SMILES"""
        perylene = Chem.MolFromSmiles("c1cc2cccc3c4cccc5cccc(c(c1)c23)c54", smi_params)
        perylene2 = Chem.MolFromSmiles("c1cc2cccc3c4cccc5cccc(c(c1)c23)c54", smi_params)
        Chem.KekulizeIfPossible(perylene2, clearAromaticFlags=True)
        for isomer in generate_optimal_aromatic_resonance_structures(perylene2):
            if perylene.HasSubstructMatch(isomer):
                break
        else:
            assert (
                False
            ), f"{Chem.MolToSmiles(perylene)} isn't isomorphic with any aromatic forms of {Chem.MolToSmiles(perylene2)}"

    def test_fused_aromatic2(self):
        """Test we can make aromatic naphthalene from both adjlist and SMILES"""
        naphthalene = Chem.MolFromSmiles("c1ccc2ccccc2c1", smi_params)
        naphthalene2 = Chem.MolFromSmiles("c1ccc2ccccc2c1", smi_params)
        Chem.KekulizeIfPossible(naphthalene2, clearAromaticFlags=True)
        for isomer in generate_optimal_aromatic_resonance_structures(naphthalene2):
            if naphthalene.HasSubstructMatch(isomer):
                break
        else:
            assert (
                False
            ), f"{Chem.MolToSmiles(naphthalene)} isn't isomorphic with any aromatic forms of {Chem.MolToSmiles(naphthalene2)}"

    def test_aromatic_resonance_structures(self):
        """Test that generate_optimal_aromatic_resonance_structures gives consistent output

        Check that we get the same resonance structure regardless of which structure we start with
        """
        # Kekulized form, radical on methyl
        struct1 = Chem.MolFromSmiles("[CH2]C1=CC=CC2=C1C=CC1=CC=CC=C12", smi_params)
        Chem.KekulizeIfPossible(struct1, clearAromaticFlags=True)

        # Kekulized form, radical on ring
        struct2 = Chem.MolFromSmiles("C=C1C=CC=C2C1=CC=C1C=C[CH]C=C12", smi_params)
        Chem.KekulizeIfPossible(struct2, clearAromaticFlags=True)

        # Aromatic form
        struct3 = Chem.MolFromSmiles("[CH2]c1cccc2c1ccc1ccccc12", smi_params)

        result1 = generate_optimal_aromatic_resonance_structures(struct1)
        result2 = generate_optimal_aromatic_resonance_structures(struct2)
        result3 = generate_optimal_aromatic_resonance_structures(struct3)

        assert len(result1) == 1
        assert len(result2) == 1
        assert len(result3) == 1

        assert result1[0].HasSubstructMatch(result2[0])
        assert result1[0].HasSubstructMatch(result3[0])

    @pytest.mark.xfail(
        reason="RDKit doesn't recognize bridged aromatic as non-aromatic."
    )
    def test_bridged_aromatic(self):
        """Test that we can handle bridged aromatics.

        This is affected by how we perceive rings. Using get_smallest_set_of_smallest_rings gives
        non-deterministic output, so using get_all_cycles_of_size allows this test to pass.

        Update: Highly-strained fused rings are no longer considered aromatic."""
        mol = Chem.MolFromSmiles("c12c3cccc1c3ccc2", smi_params)
        arom = Chem.MolFromSmiles("c1cc2c3cccc-2c3c1", smi_params)

        out = generate_resonance_structures(mol)

        assert len(out) == 1  # RDKit will treat the molecule
        assert not arom.HasSubstructMatch(out[0])

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_polycyclic_aromatic_with_non_aromatic_ring(self):
        """Test that we can make aromatic resonance structures when there is a pseudo-aromatic ring.

        This applies in cases where RDKit misidentifies one ring as aromatic, but there are other
        rings in the molecule that are actually aromatic.

        Update: Highly-strained fused rings are no longer considered aromatic."""
        mol = Chem.MolFromSmiles("c1c2cccc1C(=C)C=[C]2", smi_params)
        arom = Chem.MolFromSmiles("C=C1C=[C]c2cccc1c2", smi_params)

        out = generate_resonance_structures(mol)

        # TODO: to see what can be improved in RDKit/RDMC
        assert len(out) == 5
        assert not any(arom.HasSubstructMatch(res) for res in out)

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_polycyclic_aromatic_with_non_aromatic_ring2(self):
        """Test that we can make aromatic resonance structures when there is a pseudo-aromatic ring.

        This applies in cases where RDKit misidentifies one ring as aromatic, but there are other
        rings in the molecule that are actually aromatic."""
        mol = Chem.MolFromSmiles(
            "C=C(C1=CC2=C(C=C1C=C3)C4=CC5=CC=CC=C5C=C4C=C2)C3=C", smi_params
        )
        Chem.KekulizeIfPossible(mol)
        arom = Chem.MolFromSmiles("C=C1C=Cc2cc3c(ccc4cc5ccccc5cc43)cc2C1=C", smi_params)

        out = generate_resonance_structures(mol)

        assert len(out) == 1
        assert arom.HasSubstructMatch(out[0])

    def test_kekulize_benzene(self):
        """Test that we can kekulize benzene."""
        arom = Chem.MolFromSmiles("c1ccccc1", smi_params)
        keku = Chem.MolFromSmiles("c1ccccc1", smi_params)
        Chem.KekulizeIfPossible(keku)
        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert out[0].HasSubstructMatch(keku)

    def test_kekulize_naphthalene(self):
        """Test that we can kekulize naphthalene."""
        arom = Chem.MolFromSmiles("c1c2ccccc2ccc1", smi_params)
        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 5

    def test_kekulize_phenanthrene(self):
        """Test that we can kekulize phenanthrene."""
        arom = Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccccc12", smi_params)

        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 7

    def test_kekulize_pyrene(self):
        """Test that we can kekulize pyrene."""
        arom = Chem.MolFromSmiles("c1cc2ccc3cccc4ccc(c1)c2c34", smi_params)

        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 8

    def test_kekulize_corannulene(self):
        """Test that we can kekulize corannulene."""
        arom = Chem.MolFromSmiles("c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51", smi_params)

        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 10

    def test_kekulize_coronene(self):
        """Test that we can kekulize coronene."""
        arom = Chem.MolFromSmiles("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", smi_params)

        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 12

    def test_kekulize_bridged_aromatic(self):
        """Test that we can kekulize a bridged polycyclic aromatic species."""
        arom = Chem.MolFromSmiles("c1ccc2c3cc-3cc2c1", smi_params)

        out = generate_kekule_structure(arom)

        assert len(out) == 1
        assert not is_aromatic(out[0])

        d_bonds = 0
        for bond in out[0].GetBonds():
            if bond.GetBondType() == 2:
                d_bonds += 1

        assert d_bonds == 5

    def test_multiple_kekulized_resonance_isomers_rad(self):
        """Test we can make all resonance structures of o-cresol radical"""

        mol = Chem.MolFromSmiles("Cc1ccccc1[O]", smi_params)

        assert is_aromatic(mol), "Starting molecule should be aromatic"
        mol_list = generate_resonance_structures(mol)
        assert (
            len(mol_list) == 5
        ), f"Expected 5 resonance structures, but generated {len(mol_list)}."
        assert (
            sum([is_aromatic(mol) for mol in mol_list]) == 1
        ), "Should only have 1 aromatic resonance structure"

    def test_keep_isomorphic_structures_functions_when_true(self):
        """Test that keep_isomorphic works for resonance structure generation when True."""
        mol = Chem.MolFromSmiles("C=C[CH2]", smi_params)

        out = generate_resonance_structures(mol, keep_isomorphic=True)

        assert len(out) == 2
        assert out[0].HasSubstructMatch(out[1])
        assert not is_identical(out[0], out[1])

    def test_keep_isomorphic_structures_functions_when_false(self):
        """Test that keep_isomorphic works for resonance structure generation when False."""
        mol = Chem.MolFromSmiles("C=C[CH2]", smi_params)

        out = generate_resonance_structures(mol, keep_isomorphic=False)

        assert len(out) == 1

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_false_negative_aromaticity_perception(self):
        """Test that we obtain the correct aromatic structure for a monocyclic aromatic that RDKit mis-identifies."""
        mol = Chem.MolFromSmiles("[CH2]C=C1C=CC(=C)C=C1", smi_params)
        Chem.KekulizeIfPossible(mol)
        out = generate_resonance_structures(mol)

        aromatic = Chem.MolFromSmiles("'[CH2]c1ccc(C=C)cc1'", smi_params)

        assert len(out) == 4
        assert any([m.HasSubstructMatch(aromatic) for m in out])

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_false_negative_polycyclic_aromaticity_perception(self):
        """Test that we generate proper structures for a polycyclic aromatic that RDKit mis-identifies."""
        mol = Chem.MolFromSmiles("C=C1C=CC=C2C=C[CH]C=C12", smi_params)
        Chem.KekulizeIfPossible(mol)
        out = generate_resonance_structures(mol)

        clar = Chem.MolFromSmiles("[CH2]c1cccc2c1C=CC=C2", smi_params)

        assert len(out) == 6
        assert any([m.HasSubstructMatch(clar) for m in out])

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_false_negative_polycylic_aromaticity_perception2(self):
        """Test that we obtain the correct aromatic structure for a polycylic aromatic that RDKit mis-identifies."""
        mol = Chem.MolFromSmiles("[CH2]C=C1C=CC(=C)C2=C1C=CC=C2", smi_params)
        Chem.KekulizeIfPossible(mol)
        out = generate_resonance_structures(mol)

        aromatic = Chem.MolFromSmiles("[CH2]c1ccc(C=C)c2ccccc12", smi_params)

        assert len(out) == 7
        assert any([m.HasSubstructMatch(aromatic) for m in out])

    @pytest.mark.skip("Looks like a case catching RMG internal bugs.")
    def test_inconsistent_aromatic_structure_generation(self):
        """Test an unusual case of inconsistent aromaticity perception.

        Update: Highly-strained fused rings are no longer considered aromatic.
        That prevents the inconsistent aromatic structure for this molecule."""
        mol1 = Chem.MolFromSmiles("C1=C[C]2C=CC3=CC2=C3C1", smi_params)

        mol2 = Chem.MolFromSmiles("C1=C[C]2C=CC3=CC2=C3C1", smi_params)

        # These two slightly different adjlists should be the same structure
        assert mol1.HasSubstructMatch(mol2)

        # However, they give different resonance structures
        res1 = generate_resonance_structures(mol1)
        res2 = generate_resonance_structures(mol2)
        assert len(res1) == len(res2)

    def test_resonance_without_changing_atom_order1(self):
        """Test generating resonance structures without changing the atom order"""
        mol = Chem.MolFromSmiles("[CH2]C(C)=C(C)CO", smi_params)

        res_mols = generate_resonance_structures(mol)

        # Comparing atom symbol as its nearest neighbors
        # TODO: probably replace this with FMCS
        for res_mol in res_mols:
            for atom1, atom2 in zip(mol.GetAtoms(), res_mol.GetAtoms()):
                assert atom1.GetAtomicNum() == atom2.GetAtomicNum()
                atom1_nb = {nb.GetIdx() for nb in atom1.GetNeighbors()}
                atom2_nb = {nb.GetIdx() for nb in atom2.GetNeighbors()}
                assert atom1_nb == atom2_nb

    def test_resonance_without_changing_atom_order2(self):
        """Test generating resonance structures for aromatic molecules without changing the atom order"""
        mol = Chem.MolFromSmiles("[O]C1=CC=CC=C1", smi_params)

        res_mols = generate_resonance_structures(mol, keep_isomorphic=True)

        assert len(res_mols) == 5

        # Comparing atom symbol as its nearest neighbors
        for res_mol in res_mols:
            for atom1, atom2 in zip(mol.GetAtoms(), res_mol.GetAtoms()):
                assert atom1.GetAtomicNum() == atom2.GetAtomicNum()
                atom1_nb = {nb.GetIdx() for nb in atom1.GetNeighbors()}
                atom2_nb = {nb.GetIdx() for nb in atom2.GetNeighbors()}
                assert atom1_nb == atom2_nb


class TestClar:
    """
    Contains unit tests for Clar structure methods.
    """

    def test_clar_transformation(self):
        """Test that clarTransformation generates an aromatic ring."""
        mol = Chem.MolFromSmiles("c1ccccc1", smi_params)
        sssr = Chem.GetSymmSSSR(mol)
        _clar_transformation(mol, sssr, [1])

        assert is_aromatic(mol)

    def test_clar_optimization(self):
        """Test to ensure pi electrons are conserved during optimization"""
        mol = Chem.MolFromSmiles("C1=CC=C2C=CC=CC2=C1", smi_params)
        output = _clar_optimization(mol)

        solutions, aromatic_rings, _ = output
        for solution in solutions:
            # Remove the count of pi electrons in molecule
            # as it doesn't help us check the optimization

            # Count pi electrons in solution
            y = solution[0: len(aromatic_rings)]
            x = solution[len(aromatic_rings):]
            pi_solution = 6 * sum(y) + 2 * sum(x)

            # Check that both counts give 10 pi electrons
            assert pi_solution == 10

            # Check that we only assign 1 aromatic sextet
            assert sum(y) == 1

    def test_phenanthrene(self):
        """Test that we generate 1 Clar structure for phenanthrene."""
        smi_params = Chem.SmilesParserParams()
        smi_params.removeHs = False
        smi_params.sanitize = True
        mol = Chem.MolFromSmiles("C1=CC=C2C(C=CC3=CC=CC=C32)=C1", smi_params)
        newmol = generate_clar_structures(mol)

        # Don't sanitize mol to preverse the desired bond order
        smi_params.sanitize = False
        struct = Chem.MolFromSmiles("C1=Cc2ccccc2-c2ccccc21", smi_params)

        assert len(newmol) == 1
        assert newmol[0].HasSubstructMatch(struct)

    def test_phenalene(self):
        """Test that we generate 2 Clar structures for phenalene.

        Case where there is one non-aromatic ring."""
        smi_params = Chem.SmilesParserParams()
        smi_params.removeHs = False
        smi_params.sanitize = True
        mol = Chem.MolFromSmiles("C1=CC2=CC=CC3CC=CC(=C1)C=32")
        newmol = generate_clar_structures(mol)

        # Don't sanitize mol to preverse the desired bond order
        smi_params.sanitize = False
        struct1 = Chem.MolFromSmiles("C1=Cc2cccc3c2C(=C1)CC=C3", smi_params)
        struct2 = Chem.MolFromSmiles("C1=Cc2cccc3c2C(=C1)C=CC3", smi_params)

        assert len(newmol) == 2
        assert newmol[0].HasSubstructMatch(struct1) or newmol[0].HasSubstructMatch(
            struct2
        )
        assert newmol[1].HasSubstructMatch(struct2) or newmol[1].HasSubstructMatch(
            struct1
        )
        assert not newmol[0].HasSubstructMatch(newmol[1])

    def test_corannulene(self):
        """Test that we generate 5 Clar structures for corannulene

        Case where linear relaxation does not give an integer solution"""
        smi_params = Chem.SmilesParserParams()
        smi_params.removeHs = False
        smi_params.sanitize = True
        mol = Chem.MolFromSmiles("C1=CC2=CC=C3C=CC4=C5C6=C(C2=C35)C1=CC=C6C=C4", smi_params)
        newmol = generate_clar_structures(mol)

        smi_params.sanitize = False
        struct = Chem.MolFromSmiles("C1=Cc2ccc3c4c2C2=C1C=Cc1ccc(c-4c12)C=C3", smi_params)

        assert len(newmol) == 5
        assert newmol[0].HasSubstructMatch(struct)
        assert newmol[1].HasSubstructMatch(struct)
        assert newmol[2].HasSubstructMatch(struct)
        assert newmol[3].HasSubstructMatch(struct)
        assert newmol[4].HasSubstructMatch(struct)

    @pytest.mark.xfail(reason="This test is intended to detect RDKit failtuire in RMG.")
    def test_exocyclic_db(self):
        """Test that Clar structure generation doesn't modify exocyclic double bonds

        Important for cases where RDKit considers rings to be aromatic by counting pi-electron contributions
        from exocyclic double bonds, while they don't actually contribute to aromaticity
        """

        mol = Chem.MolFromSmiles("C=C1C=CC=CC1=C")
        newmol = generate_clar_structures(mol)

        assert len(newmol) == 0

    def test_sulfur_triple_bond(self):
        """
        Test the prevention of S#S formation through the find_lone_pair_multiplebond_paths and
        find_adj_lone_pair_multiple_bond_delocalization_paths
        """
        mol = Chem.MolFromSmiles("S1SSS1", smi_params)
        mol_list = generate_resonance_structures(mol, filter_structures=False)
        # Originally RMG has 10 structures, where two of them have +2 charge
        # RDMC avoid 2+ from generation, therefore use 8 as reference value
        assert len(mol_list) == 8
