#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module includes an RDKit-based rewrite of the RMG resonance pathfinder.
"""

from typing import List

from rdkit import Chem


# Relevant atom types/changes in the resonance algorithm
# 1. radical
# lone pair <=> electron + bond electron
# 2. lose lone pair gain charge gain bond (e.g., [::N-]= -> [:N]# )
# 3. gain lone pair lose charge lose bond (e.g., [NH+]# -> [:NH]= )
# lone pair <=> electron + radical electron
# 4. lose lone pair gain charge gain radical
# 5. gain lone pair lose charge lose radical
# lone pair <=> radical electron + bond electron
# 6. lose lone pair gain radical gain bond
# 7. gain lone pair lose radical lose bond
# bond <=> lone pair of the other atom
# 8. gain charge lose bond
# 9. lose charge gain bond
# bond electron <=> radical electron
# 10. gain radical lose bond
# 11. lose radical gain bond

query_radical = {
    "C": "Cv{1-3}",
    "N": "Nv{1-2}",
    "O": "Ov1",
    "Si": "Siv{1-3}",
    "P": "Pv{1-2},Pv4",
    "S": "Sv1,Sv{3-5}",
}

query_lose_lone_pair_gain_charge_gain_bond = {
    # lose one electron of the pair (charge +1) and homolytically form a bond
    # # A few entries are excluded as trying to avoid atom with ++ or -- charge
    "C": "C-v{1-3},Cv{1-2}",
    "N": "N-v{1-2},Nv{1-3}",
    "O": "O-v1,Ov{1-2}",
    "P": "P-v{1-4},Pv{1-3}",
    "S": "S-v{1-5},Sv{1-4}",
}

query_gain_lone_pair_lose_charge_lose_bond = {
    #  homolytically break a bond and gain one electron (charge -1) to form a lone pair
    # A few entries are excluded as trying to avoid atom with ++ or -- charge
    "C": "C+v2X1,C+v3X{1-2},Cv2X1,Cv3X{1-2},Cv4X{1-3}",
    "N": "N+v2X1,N+v3X{1-2},N+v4X{1-3},Nv2X1,Nv3X{1-2}",
    "O": "O+v2X1,O+v3X{1-2},Ov2X1",
    "P": "P+v2X1,P+v3X{1-2},P+v4X{1-3},Pv2X1,Pv3X{1-2},Pv4X{1-3},Pv5X{1-4}",
    "S": "S+v2X1,S+v3X{1-2},S+v4X{1-3},S+v5X{1-4},Sv2X1,Sv3X{1-2},Sv4X{1-3},Sv5X{1-4},Sv6X{1-5}",
}

# The following is derived manually
# It is found to be the same set as query_lose_lone_gain_charge_pair_gain_bond
# This is because the only difference is whether the electron is used to
# form a bond or leave as a radical electron
# query_gain_radical_gain_charge_lose_lone_pair = {
#     # lose one electron from a lone pair (charge + 1) and form a radical
#     "C": "C-v{1-3},Cv{1-2}",
#     "N": "N-v{1-2},Nv{1-3}",
#     "O": "O-v1,Ov{1-2}",
#     "P": "P-v{1-4},Pv{1-3}",
#     "S": "S-v{1-5},Sv{1-4}",
# }
query_lose_lone_pair_gain_charge_gain_radical = (
    query_lose_lone_pair_gain_charge_gain_bond
)

query_gain_lone_pair_lose_charge_lose_radical = {
    # gain an electron (charge - 1) and form a lone pair with one radical electron
    "C": "C+v{1-2},Cv{1-3}",
    "N": "N+v{1-3},Nv{1-2}",
    "O": "O+v{1-2},Ov1",
    "P": "P+v{1-3},Pv{1-4}",
    "S": "S+v{1-4},Sv{1-5}",
}

query_lose_lone_pair_gain_radical_gain_bond = {
    "C": "C+v1,C-v1,Cv{1-2}",
    "N": "N+v{1-2},Nv1",
    "O": "O+v1",
    "P": "P+v{1-2},P-v{1-4},Pv{1-3}",
    "S": "S+v{1-3},S-v{1-3},Sv{1-4}",
}

query_gain_lone_pair_lose_radical_lose_bond = {
    "C": "C+v2X1,C-v2X1,Cv2X1,Cv3X{1-2}",
    "N": "N+v2X1,N+v3X{1-2},Nv2X1",
    "O": "O+v2X1",
    "P": "P+v2X1,P+v3X{1-2},P-v2X1,P-v3X{1-2},P-v4X{1-3},P-v5X{1-4},Pv2X1,Pv3X{1-2},Pv4X{1-3}",
    "S": "S+v2X1,S+v3X{1-2},S+v4X{1-3},S-v2X1,S-v3X{1-2},S-v4X{1-3},Sv2X1,Sv3X{1-2},Sv4X{1-3},Sv5X{1-4}",
}

query_gain_charge_lose_bond = {
    # E.g., [:N]# -> [::N-]=
    "C": "C-v2X1,C-v3X{1-2},Cv2X1,Cv3X{1-2},Cv4X{1-3}",
    "N": "N-v2X1,Nv2X1,Nv3X{1-2}",
    "O": "Ov2X1",
    "P": "P-v2X1,P-v3X{1-2},P-v4X{1-3},P-v5X{1-4},P-v6X{1-5},Pv2X1,Pv3X{1-2},Pv4X{1-3},Pv5X{1-4}",
    "S": "S-v2X1,S-v3X{1-2},S-v4X{1-3},S-v5X{1-4},Sv2X1,Sv3X{1-2},Sv4X{1-3},Sv5X{1-4},Sv6X{1-5}",
}

query_lose_charge_gain_bond = {
    # E.g., [CH2+]- -> [CH2]=
    "C": "C+v{1-3},Cv{1-2}",
    "N": "N+v{1-2},Nv1",
    "O": "O+v1",
    "P": "P+v{1-4},Pv{1-5}",
    "S": "S+v{1-5},Sv{1-4}",
}

query_gain_radical_lose_bond = {
    # E.g., [CH2]= -> [CH2.]-
    "C": "C+v2X1,C+v3X{1-2},C-v2X1,C-v3X{1-2},Cv2X1,Cv3X{1-2},Cv4X{1-3}",
    "N": "N+v2X1,N+v3X{1-2},N+v3X{1-3},N-v2X1,Nv2X1,Nv3X{1-2}",
    "O": "O+v2X1,O+v3X{1-2},Ov2X1",
    "P": (
        "P+v2X1,P+v3X{1-2},P+v4X{1-3},P-v2X1,P-v3X{1-2},P-v4X{1-3},P-v5X{1-4},P-v6X{1-5},"
        "Pv2X1,Pv3X{1-2},Pv4X{1-3},Pv5X{1-4}"
    ),
    "S": (
        "S+v2X1,S+v3X{1-2},S+v4X{1-3},S+v5X{1-4},S-v2X1,S-v3X{1-2},S-v4X{1-3},S-v5X{1-4},"
        "Sv2X1,Sv3X{1-2},Sv4X{1-3},Sv5X{1-4},Sv6X{1-5}"
    ),
}

query_lose_radical_gain_bond = {
    # E.g., [CH2.]- -> [CH2]=
    "C": "C+v{1-2},C-v{1-2},Cv{1-3}",
    "N": "N+v{1-3},N-v1,Nv{1-2}",
    "O": "O+v{1-2},Ov1",
    "P": "P+v{1-3},P-v{1-5},Pv{1-4}",
    "S": "S+v{1-4},S-v{1-4},Sv{1-5}",
}

# More generic queries
hetero_lose_lone_pair_gain_charge_gain_bond = ",".join(
    [query_lose_lone_pair_gain_charge_gain_bond[elem] for elem in "NOPS"]
)
hetero_gain_lone_pair_lose_charge_lose_bond = ",".join(
    [query_gain_lone_pair_lose_charge_lose_bond[elem] for elem in "NOPS"]
)
hetero_lose_lone_pair_gain_charge_gain_radical = ",".join(
    [query_lose_lone_pair_gain_charge_gain_radical[elem] for elem in "NOPS"]
)
hetero_gain_lone_pair_lose_charge_lose_radical = ",".join(
    [query_gain_lone_pair_lose_charge_lose_radical[elem] for elem in "NOPS"]
)
hetero_lose_lone_pair_gain_radical_gain_bond = ",".join(
    [query_lose_lone_pair_gain_radical_gain_bond[elem] for elem in "NOPS"]
)
hetero_gain_lone_pair_lose_radical_lose_bond = ",".join(
    [query_gain_lone_pair_lose_radical_lose_bond[elem] for elem in "NOPS"]
)
hetero_gain_charge_lose_bond = ",".join(
    [query_gain_charge_lose_bond[elem] for elem in "NOPS"]
)
hetero_lose_charge_gain_bond = ",".join(
    [query_lose_charge_gain_bond[elem] for elem in "NOPS"]
)
hetero_gain_radical_lose_bond = ",".join(
    [query_gain_radical_lose_bond[elem] for elem in "NOPS"]
)
hetero_lose_radical_gain_bond = ",".join(
    [query_lose_radical_gain_bond[elem] for elem in "NOPS"]
)

atom_radical = ",".join(query_radical.values())
atom_lose_lone_pair_gain_charge_gain_bond = ",".join(
    query_lose_lone_pair_gain_charge_gain_bond.values()
)
atom_gain_lone_pair_lose_charge_lose_bond = ",".join(
    query_gain_lone_pair_lose_charge_lose_bond.values()
)
atom_lose_lone_pair_gain_charge_gain_radical = ",".join(
    query_lose_lone_pair_gain_charge_gain_radical.values()
)
atom_gain_lone_pair_lose_charge_lose_radical = ",".join(
    query_gain_lone_pair_lose_charge_lose_radical.values()
)
atom_lose_lone_pair_gain_radical_gain_bond = ",".join(
    query_lose_lone_pair_gain_radical_gain_bond.values()
)
atom_gain_lone_pair_lose_radical_lose_bond = ",".join(
    query_gain_lone_pair_lose_radical_lose_bond.values()
)
atom_gain_charge_lose_bond = ",".join(query_gain_charge_lose_bond.values())
atom_lose_charge_gain_bond = ",".join(query_lose_charge_gain_bond.values())
atom_gain_radical_lose_bond = ",".join(query_gain_radical_lose_bond.values())
atom_lose_radical_gain_bond = ",".join(query_lose_radical_gain_bond.values())

# Defined resonance templates based on the above-defined atom types
ALLYL_RADICAL_TEMPLATE = Chem.MolFromSmarts(f"[{atom_radical}:0]-,=[*:1]=,#[*:2]")

LONE_PAIR_MULTIPLE_BOND_TEMPLATE = Chem.MolFromSmarts(
    f"[{hetero_lose_lone_pair_gain_charge_gain_bond}:1]-,=[*:2]=,#[{atom_gain_lone_pair_lose_charge_lose_bond}:3]"
)

ADJ_LONE_PAIR_RADICAL_TEMPLATE = Chem.MolFromSmarts(
    f"[{atom_gain_lone_pair_lose_charge_lose_radical}:1]~[{atom_lose_lone_pair_gain_charge_gain_radical}:2]"
)

ADJ_LONE_PAIR_GAIN_BOND = Chem.MolFromSmarts(
    f"[{hetero_lose_lone_pair_gain_charge_gain_bond}:1]-,=[{atom_lose_charge_gain_bond}:2]"
)

ADJ_LONE_PAIR_LOSE_BOND = Chem.MolFromSmarts(
    f"[{hetero_gain_lone_pair_lose_charge_lose_bond}:1]=,#[{atom_gain_charge_lose_bond}:2]"
)

ADJ_LONE_PAIR_RADICAL_GAIN_BOND = Chem.MolFromSmarts(
    f"[{hetero_lose_lone_pair_gain_radical_gain_bond}:1]-,=[{atom_lose_radical_gain_bond}:2]"
)

ADJ_LONE_PAIR_RADICAL_LOSE_BOND = Chem.MolFromSmarts(
    f"[{hetero_gain_lone_pair_lose_radical_lose_bond}:1]=,#[{atom_gain_radical_lose_bond}:2]"
)


def find_paths_with_template(
    mol: "RWMol",
    template: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find the delocalization paths for allyl radicals.
    """
    return list(
        mol.GetSubstructMatches(
            template,
            uniquify=False,
            maxMatches=max_matches,
        )
    )


def find_allyl_delocalization_paths(
    mol: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find all the delocalization paths allyl to the radical center.
    """
    return find_paths_with_template(
        mol,
        ALLYL_RADICAL_TEMPLATE,
        max_matches=max_matches,
    )


def find_lone_pair_multiple_bond_paths(
    mol: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find all the delocalization paths between lone electron pair and multiple bond in a 3-atom system.

    Examples:

    - N2O (N#[N+][O-] <-> [N-]=[N+]=O)
    - Azide (N#[N+][NH-] <-> [N-]=[N+]=N <-> [N-2][N+]#[NH+])
    - N#N group on sulfur (O[S-](O)[N+]#N <-> OS(O)=[N+]=[N-] <-> O[S+](O)#[N+][N-2])
    - N[N+]([O-])=O <=> N[N+](=O)[O-], these structures are isomorphic but not identical, this transition is
      important for correct degeneracy calculations
    """
    return find_paths_with_template(
        mol,
        LONE_PAIR_MULTIPLE_BOND_TEMPLATE,
        max_matches=max_matches,
    )


def find_adj_lone_pair_radical_delocalization_paths(
    mol: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find all the delocalization paths of lone electron pairs next to the radical center.
    Used to generate resonance isomers in adjacent N/O/S atoms.

    The radical site (atom1) could be either:

    - `N u1 p0`, eg O=[N.+][:::O-]
    - `N u1 p1`, eg R[:NH][:NH.]
    - `O u1 p1`, eg [:O.+]=[::N-]; not allowed when adjacent to another O atom
    - `O u1 p2`, eg O=N[::O.]; not allowed when adjacent to another O atom
    - `S u1 p0`, eg O[S.+]([O-])=O
    - `S u1 p1`, eg O[:S.+][O-]
    - `S u1 p2`, eg O=N[::S.]
    - any of the above with more than 1 radical where possible

    The non-radical site (atom2) could respectively be:

    - `N u0 p1`
    - `N u0 p2`
    - `O u0 p2`
    - `O u0 p3`
    - `S u0 p1`
    - `S u0 p2`
    - `S u0 p3`

    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    The bond between the sites does not have to be single, e.g.: [:O.+]=[::N-] <=> [::O]=[:N.]
    """
    return find_paths_with_template(
        mol,
        ADJ_LONE_PAIR_RADICAL_TEMPLATE,
        max_matches=max_matches,
    )


def find_adj_lone_pair_multiple_bond_delocalization_paths(
    mol: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find all the delocalization paths of atom1 which either

    - Has a lonePair and is bonded by a single/double bond (e.g., [::NH-]-[CH2+], [::N-]=[CH+]) -- direction 1
    - Can obtain a lonePair and is bonded by a double/triple bond (e.g., [:NH]=[CH2], [:N]#[CH]) -- direction 2

    Giving the following resonance transitions, for example:

    - [::NH-]-[CH2+] <=> [:NH]=[CH2]
    - [:N]#[CH] <=> [::N-]=[CH+]
    - other examples: S#N, N#[S], O=S([O])=O

    Direction "1" is the direction <increasing> the bond order as in [::NH-]-[CH2+] <=> [:NH]=[CH2]
    Direction "2" is the direction <decreasing> the bond order as in [:NH]=[CH2] <=> [::NH-]-[CH2+]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    (In direction 1 atom1 <losses> a lone pair, in direction 2 atom1 <gains> a lone pair)
    """
    return [
        (a1, a2, 1)  # the last bit is the direction
        for a1, a2 in find_paths_with_template(
            mol,
            ADJ_LONE_PAIR_GAIN_BOND,
            max_matches=max_matches,
        )
    ] + [
        (a1, a2, 2)  # the last bit is the direction
        for a1, a2 in find_paths_with_template(
            mol,
            ADJ_LONE_PAIR_LOSE_BOND,
            max_matches=max_matches,
        )
    ]


def find_adj_lone_pair_radical_multiple_bond_delocalization_paths(
    mol: "RWMol",
    max_matches: int = 1000,
) -> List[tuple]:
    """
    Find all the delocalization paths of atom1 which either

    - Has a lonePair and is bonded by a single/double bond to a radical atom (e.g., [::N]-[.CH2])
    - Can obtain a lonePair, has a radical, and is bonded by a double/triple bond (e.g., [:N.]=[CH2])

    Giving the following resonance transitions, for example:

    - [::N]-[.CH2] <=> [:N.]=[CH2]
    - O[:S](=O)[::O.] <=> O[S.](=O)=[::O]

    Direction "1" is the direction <increasing> the bond order as in [::N]-[.CH2] <=> [:N.]=[CH2]
    Direction "2" is the direction <decreasing> the bond order as in [:N.]=[CH2] <=> [::N]-[.CH2]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    (In direction 1 atom1 <losses> a lone pair, gains a radical, and atom2 looses a radical.
    In direction 2 atom1 <gains> a lone pair, looses a radical, and atom2 gains a radical)
    """
    return [
        (a1, a2, 1)  # the last bit is the direction
        for a1, a2 in find_paths_with_template(
            mol,
            ADJ_LONE_PAIR_RADICAL_GAIN_BOND,
            max_matches=max_matches,
        )
    ] + [
        (a1, a2, 2)  # the last bit is the direction
        for a1, a2 in find_paths_with_template(
            mol,
            ADJ_LONE_PAIR_RADICAL_LOSE_BOND,
            max_matches=max_matches,
        )
    ]


# def find_N5dc_radical_delocalization_paths(atom1)
# It is included in the original version
# But the relevant resonance transformation is recoverable
# by the above methods.
# Removed in the rewriting.
