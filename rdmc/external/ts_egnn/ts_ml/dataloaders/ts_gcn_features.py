from rdkit import Chem
from typing import List, Tuple, Union
from ..dataloaders.features import onek_encoding_unk


ATOMIC_SYMBOLS = ['H', 'C', 'N', 'O']
ATOM_FEATURES = {
    'atomic_num': ATOMIC_SYMBOLS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 7


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_num']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return fbond


class MolGraph:

    def __init__(self, mols, no_ts=False):

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
        tD_r = Chem.GetDistanceMatrix(r_mol)
        tD_p = Chem.GetDistanceMatrix(p_mol)
        D_r = Chem.Get3DDistanceMatrix(r_mol)
        D_p = Chem.Get3DDistanceMatrix(p_mol)
        if not no_ts:
            D_ts = Chem.Get3DDistanceMatrix(ts_mol)

        # temporary featurization
        for a1 in range(n_atoms):

            # Node features
            self.f_atoms.append(atom_features(r_mol.GetAtomWithIdx(a1)))

            # Edge features
            for a2 in range(a1 + 1, n_atoms):
                # fully connected graph
                self.edge_index.extend([(a1, a2), (a2, a1)])

                # for now, naively include both reac and prod
                b1_feats = [D_r[a1][a2], D_p[a1][a2]]
                b2_feats = [D_r[a2][a1], D_p[a2][a1]]

                r_bond = r_mol.GetBondBetweenAtoms(a1, a2)
                b1_feats.extend(bond_features(r_bond))
                b2_feats.extend(bond_features(r_bond))

                p_bond = p_mol.GetBondBetweenAtoms(a1, a2)
                b1_feats.extend(bond_features(p_bond))
                b2_feats.extend(bond_features(p_bond))

                self.f_bonds.append(b1_feats)
                self.f_bonds.append(b2_feats)
                if not no_ts:
                    self.y.extend([D_ts[a1][a2], D_ts[a2][a1]])
