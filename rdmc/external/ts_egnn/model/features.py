from rdkit import Chem
from rdmc.mol import RDKitMol
from rdmc.ts import NaiveAlign, get_formed_and_broken_bonds
from .align import reset_pmol
import numpy as np
from typing import List, Union
from scipy.spatial.transform import Rotation

ATOMIC_SYMBOLS = ['H', 'C', 'N', 'O']
BOND_FDIM = 5
RXN_BOND_FDIM = 5


def onek_encoding_unk(value: int, choices: List[Union[str, int]]) -> List[int]:
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom):
    features = onek_encoding_unk(atom.GetSymbol(), ATOMIC_SYMBOLS)
    return features


def bond_features(bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC
        ]
    return fbond


def reaction_bond_features(r_bond, p_bond):
    if r_bond is None and p_bond is None:
        return [0] * RXN_BOND_FDIM
    elif r_bond is None:
        return [0] + [1] * (RXN_BOND_FDIM - 1)  # bond broken
    elif p_bond is None:
        return [0] + [0] + [1] * (RXN_BOND_FDIM - 2)  # bond formed

    # use % 5.25 to make aromatic = 1.5
    r_bt = int(r_bond.GetBondType()) % 5.25
    p_bt = int(p_bond.GetBondType()) % 5.25
    if r_bt == p_bt:
        return [1] * RXN_BOND_FDIM  # no change to existing bond
    elif p_bt > r_bt:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


def get_any_bonds(r_mol, p_mol):
    r_bonds = [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in r_mol.GetBonds()]
    p_bonds = [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in p_mol.GetBonds()]
    bonds = list(set(r_bonds) | set(p_bonds))
    return sorted(bonds)


class MolGraph:

    def __init__(self, mols, prod_feat):

        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.edge_index = []  # list of tuples indicating pairwise atoms
        self.y = []

        # extract reactant, ts, product
        r_mol, ts_mol, p_mol = mols

        # compute properties with rdkit (only works if dataset is clean)
        r_mol.UpdatePropertyCache()
        p_mol.UpdatePropertyCache()

        # fake the number of "atoms" if we are collapsing substructures
        n_atoms = r_mol.GetNumAtoms()

        # topological and 3d distance matrices
        D_r = Chem.Get3DDistanceMatrix(r_mol)
        if prod_feat == 'dist':
            D_p = Chem.Get3DDistanceMatrix(p_mol)
        elif prod_feat == 'adj':
            D_p = Chem.GetAdjacencyMatrix(p_mol)

        # any bonds between reactants and products
        self.bonded_index = get_any_bonds(r_mol, p_mol)

        # featurization
        for a1 in range(n_atoms):

            # Node features
            self.f_atoms.append(atom_features(r_mol.GetAtomWithIdx(a1)))

            # Edge features
            for a2 in range(a1 + 1, n_atoms):
                # fully connected graph
                self.edge_index.extend([(a1, a2), (a2, a1)])

                # ONLY INCLUDE PRODUCT DISTANCES SINCE REACTANT APPENDED IN NETWORK
                # b1_feats = [D_r[a1][a2], D_p[a1][a2]]
                # b2_feats = [D_r[a2][a1], D_p[a2][a1]]
                b1_feats = [D_p[a1][a2]]
                b2_feats = [D_p[a2][a1]]

                r_bond = r_mol.GetBondBetweenAtoms(a1, a2)
                b1_feats.extend(bond_features(r_bond))
                b2_feats.extend(bond_features(r_bond))

                p_bond = p_mol.GetBondBetweenAtoms(a1, a2)
                b1_feats.extend(bond_features(p_bond))
                b2_feats.extend(bond_features(p_bond))

                rxn_feats = reaction_bond_features(r_bond, p_bond)
                b1_feats.extend(rxn_feats)
                b2_feats.extend(rxn_feats)

                self.f_bonds.append(b1_feats)
                self.f_bonds.append(b2_feats)


def shuffle_mols(mols):

    r_mol, ts_mol, p_mol = mols
    if np.random.random_sample(1) < 0.5:
        return r_mol, ts_mol, p_mol
    else:
        return p_mol, ts_mol, r_mol


def find_similar_mol(mols):

    r_mol, ts_mol, p_mol = mols
    r_rdmc, ts_rdmc, p_rdmc = RDKitMol.FromMol(r_mol), RDKitMol.FromMol(ts_mol), RDKitMol.FromMol(p_mol)

    _, rmsd_r = Rotation.align_vectors(r_rdmc.GetPositions(), ts_rdmc.GetPositions())
    _, rmsd_p = Rotation.align_vectors(p_rdmc.GetPositions(), ts_rdmc.GetPositions())
    if rmsd_r < rmsd_p:
        return r_mol, ts_mol, p_mol
    else:
        return p_mol, ts_mol, r_mol


def realistic_mol_prep(mols):
    r_mol, ts_mol, p_mol = mols
    r_rdmc, p_rdmc = RDKitMol.FromMol(r_mol), RDKitMol.FromMol(p_mol)
    if len(r_rdmc.GetMolFrags()) == 2:
        r_rdmc = align_reactant_fragments(r_rdmc, p_rdmc)
    p_mol_new = reset_pmol(r_rdmc, p_rdmc)  # reconfigure pmol as if starting from SMILES
    # p_mol_new = optimize_rotatable_bonds(r_rdmc, p_mol_new)  # optimize rotatable bonds
    return r_rdmc.ToRWMol(), ts_mol, p_mol_new.ToRWMol()


def align_reactant_fragments(r_rdmc, p_rdmc):
    formed_bonds, broken_bonds = get_formed_and_broken_bonds(r_rdmc, p_rdmc)
    naive_align = NaiveAlign.from_complex(r_rdmc, formed_bonds, broken_bonds)
    r_rdmc_naive_align = r_rdmc.Copy()
    r_rdmc_naive_align.SetPositions(naive_align())
    return r_rdmc_naive_align
