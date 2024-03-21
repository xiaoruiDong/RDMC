#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module includes an RDKit-based rewrite of the RMG resonance pathfinder.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple, Union

from rdmc.rdtools.resonance.utils import (
    decrement_order,
    decrement_radical,
    get_lone_pair,
    has_empty_orbitals,
    increment_order,
    increment_radical,
    sanitize_resonance_mol,
)
from rdkit import Chem


logger = logging.getLogger(__name__)

# This is the implicit charge constraint embedded in the following path finding templates
# I.e., for an query lose charge, the atom must have a charge >= 0, so that the resulting
# charge is >= -1. The constraint is explicitly defined here for clarity, and easy for
# other part to import.
CHARGE_UPPER_LIMIT = 1
CHARGE_LOWER_LIMIT = -1


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


def transform_pre_and_post_process(fun):
    """
    A decorator for resonance structure generation functions that require a radical.

    Returns an empty list if the input molecule is not a radical.
    """

    def wrapper(mol, path, *args, **kwargs):
        structure = Chem.RWMol(mol, True)
        try:
            fun(structure, path, *args, **kwargs)
            sanitize_resonance_mol(structure)
        except BaseException as e:  # cannot make the change
            class_name = fun.__qualname__.split(".")[0]
            logger.debug(
                f"Cannot transform path {path} "
                f"in `{class_name}.{fun.__name__}`."
                f"\nGot: {e}"
            )
        else:
            return structure

    return wrapper


class PathFinderRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(some_class):
            cls._registry[name] = some_class
            return some_class

        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)


class PathFinder(ABC):
    @classmethod
    def find(
        cls,
        mol: Union["RWMol", "RDKitMol"],
        max_matches: int = 1000,
    ) -> Set[Tuple[int]]:
        """
        Find the paths for according to the template of class.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            max_matches (int): The maximum number of matches to return. Defaults to 1000.

        Returns:
            set: A set of tuples containing the atom indices of the paths.
        """
        return set(
            mol.GetSubstructMatches(
                cls.template,
                uniquify=False,
                maxMatches=max_matches,
            )
        )

    @abstractmethod
    def verify(mol, path) -> bool:
        """
        Abstract method that should return True if the path is valid, False otherwise.

        This needs to be implemented by subclasses.
        """

    @abstractmethod
    def transform(mol, path) -> "Mol":
        """
        Abstract method that should return the transformed molecule based on the provided path.

        This needs to be implemented by subclasses.
        """


@PathFinderRegistry.register("allyl_radical")
class AllylRadicalPathFinder(PathFinder):
    """
    A finder to find all the delocalization paths allyl to the radical center.
    """

    template = Chem.MolFromSmarts(f"[{atom_radical}:0]-,=[*:1]=,#[*:2]")

    reverse = "AllylRadicalPathFinder"
    reverse_abbr = "allyl_radical"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the allyl delocalization path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The allyl delocalization path to verify.

        Returns:
            bool: True if the allyl delocalization path is valid, False otherwise.
        """
        return mol.GetAtomWithIdx(path[0]).GetNumRadicalElectrons() > 0

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of ally delocalization.

        Args:
            mol (RDKitMol): The molecule to be processed.
            path (tuple): The path of allyl delocalization.

        Returns:
            Mol: if the transformation is successful, return the transformed molecule;
                otherwise, return ``None``.
        """
        a1_idx, a2_idx, a3_idx = path

        decrement_radical(mol.GetAtomWithIdx(a1_idx))
        increment_radical(mol.GetAtomWithIdx(a3_idx))
        increment_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        decrement_order(mol.GetBondBetweenAtoms(a2_idx, a3_idx))

        return mol


@PathFinderRegistry.register("lone_pair_multiple_bond")
class LonePairMultipleBondPathFinder(PathFinder):
    """
    A finder to find all the delocalization paths between lone electron pair and multiple bond in a 3-atom system.

    Examples:

    - N2O (N#[N+][O-] <-> [N-]=[N+]=O)
    - Azide (N#[N+][NH-] <-> [N-]=[N+]=N <-> [N-2][N+]#[NH+])
    - N#N group on sulfur (O[S-](O)[N+]#N <-> OS(O)=[N+]=[N-] <-> O[S+](O)#[N+][N-2])
    - N[N+]([O-])=O <=> N[N+](=O)[O-], these structures are isomorphic but not identical, this transition is
      important for correct degeneracy calculations
    """

    template = Chem.MolFromSmarts(
        f"[{hetero_lose_lone_pair_gain_charge_gain_bond}:1]-,=[*:2]=,#[{atom_gain_lone_pair_lose_charge_lose_bond}:3]"
    )

    reverse = "LonePairMultipleBondPathFinder"
    reverse_abbr = "lone_pair_multiple_bond"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the lone pair multiple bond path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The lone pair multiple bond path to verify.

        Returns:
            bool: True if the lone pair multiple bond path is valid, False otherwise.
        """
        a1_idx, a2_idx, a3_idx = path
        a1 = mol.GetAtomWithIdx(a1_idx)
        a2 = mol.GetAtomWithIdx(a2_idx)
        a3 = mol.GetAtomWithIdx(a3_idx)
        return (
            not (
                a1.GetAtomicNum() == a2.GetAtomicNum() == 16
            )  # RMG algorithm avoid a1 and a2 are both S
            and a1.GetFormalCharge() < CHARGE_UPPER_LIMIT
            and a3.GetFormalCharge() > CHARGE_LOWER_LIMIT
            and get_lone_pair(a1) > 0
        )

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of lone pair multiple bond.

        Args:
            mol (RDKitMol): The molecule to be processed.
            path (tuple): The path of lone pair multiple bond.

        Returns:
            Mol: if the transformation is successful, return the transformed molecule;
        """
        a1_idx, a2_idx, a3_idx = path
        a1, a3 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a3_idx)

        increment_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        decrement_order(mol.GetBondBetweenAtoms(a2_idx, a3_idx))
        a1.SetFormalCharge(a1.GetFormalCharge() + 1)
        a3.SetFormalCharge(a3.GetFormalCharge() - 1)

        return mol


@PathFinderRegistry.register("adj_lone_pair_radical")
class AdjacentLonePairRadicalPathFinder(PathFinder):
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

    template = Chem.MolFromSmarts(
        f"[{atom_gain_lone_pair_lose_charge_lose_radical}:1]~[{atom_lose_lone_pair_gain_charge_gain_radical}:2]"
    )

    reverse = "AdjacentLonePairRadicalPathFinder"
    reverse_abbr = "adj_lone_pair_radical"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the adjacent lone pair radical delocalization path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The adjacent lone pair radical delocalization path to verify.

        Returns:
            bool: True if the adjacent lone pair radical delocalization path is valid, False otherwise.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        return (
            a1.GetFormalCharge() > CHARGE_LOWER_LIMIT
            and a2.GetFormalCharge() < CHARGE_UPPER_LIMIT
            and a1.GetNumRadicalElectrons() > 0
            and get_lone_pair(a2) > 0
        )

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of adjacent lone pair radical delocalization.

        Args:
            mol (RDKitMol): The molecule to be processed.
            path (tuple): The path of adjacent lone pair radical delocalization.

        Returns:
            Mol: if the transformation is successful, return the transformed molecule;
                otherwise, return ``None``.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        decrement_radical(a1)
        increment_radical(a2)
        a1.SetFormalCharge(a1.GetFormalCharge() - 1)
        a2.SetFormalCharge(a2.GetFormalCharge() + 1)

        return mol


@PathFinderRegistry.register("forward_adj_lone_pair_multiple_bond")
class ForwardAdjacentLonePairMultipleBondPathFinder(PathFinder):
    """
    Find all the delocalization paths of atom1 that has a lonePair and is bonded by a single/double bond
    e.g., [::NH-]-[CH2+] <=> [:NH]=[CH2].
    """

    template = Chem.MolFromSmarts(
        f"[{hetero_lose_lone_pair_gain_charge_gain_bond}:1]-,=[{atom_lose_charge_gain_bond}:2]"
    )

    reverse = "ReverseAdjacentLonePairMultipleBondPathFinder"
    reverse_abbr = "reverse_adj_lone_pair_multiple_bond"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """ """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)
        return (
            not (
                a1.GetAtomicNum() == a2.GetAtomicNum() == 16
                and mol.GetBondBetweenAtoms(a1_idx, a2_idx).GetBondType() == 2
            )  # RMG have this to prevent S#S. This may be better added to the template
            and a1.GetFormalCharge() < CHARGE_UPPER_LIMIT
            and a2.GetFormalCharge() > CHARGE_LOWER_LIMIT
            and get_lone_pair(a1) > 0
            and has_empty_orbitals(a2)
        )

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of adjacent lone pair multiple bond delocalization.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        increment_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        a1.SetFormalCharge(a1.GetFormalCharge() + 1)
        a2.SetFormalCharge(a2.GetFormalCharge() - 1)

        return mol


@PathFinderRegistry.register("reverse_adj_lone_pair_multiple_bond")
class ReverseAdjacentLonePairMultipleBondPathFinder(PathFinder):
    """
    Find all the delocalization paths of atom1 which either can obtain a lonePair and
    is bonded by a double/triple bond (e.g., [:NH]=[CH2], [:N]#[CH]). For example, [:NH]=[CH2] <=> [::NH-]-[CH2+]
    """

    template = Chem.MolFromSmarts(
        f"[{hetero_gain_lone_pair_lose_charge_lose_bond}:1]=,#[{atom_gain_charge_lose_bond}:2]"
    )

    reverse = "ForwardAdjacentLonePairMultipleBondPathFinder"
    reverse_abbr = "forward_adj_lone_pair_multiple_bond"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the adjacent lone pair multiple bond delocalization path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The adjacent lone pair multiple bond delocalization path to verify.

        Returns:
            bool: True if the adjacent lone pair multiple bond delocalization path is valid, False otherwise.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        return (
            a1.GetFormalCharge() > CHARGE_LOWER_LIMIT
            and a2.GetFormalCharge() < CHARGE_UPPER_LIMIT
        )

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of adjacent lone pair multiple bond delocalization.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        decrement_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        a1.SetFormalCharge(a1.GetFormalCharge() - 1)
        a2.SetFormalCharge(a2.GetFormalCharge() + 1)

        return mol


@PathFinderRegistry.register("forward_adj_lone_pair_radical_multiple_bond")
class ForwardAdjacentLonePairRadicalMultipleBondPathFinder(PathFinder):
    """
    Find all the delocalization paths of atom1 which has a lonePair and is bonded
    by a single/double bond to a radical atom. For example: [::N]-[.CH2] <=> [:N.]=[CH2]
    """

    template = Chem.MolFromSmarts(
        f"[{hetero_lose_lone_pair_gain_radical_gain_bond}:1]-,=[{atom_lose_radical_gain_bond}:2]"
    )

    reverse = "ReverseAdjacentLonePairRadicalMultipleBondPathFinder"
    reverse_abbr = "reverse_adj_lone_pair_radical_multiple_bond"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the adjacent lone pair radical multiple bond delocalization path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The ÷÷÷÷≥≥adjacent lone pair multiple bond delocalization path to verify.

        Returns:
            bool: True if the adjacent lone pair multiple bond delocalization path is valid, False otherwise.
        """
        a1_idx, a2_idx = path
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)

        return (
            a2.GetNumRadicalElectrons() > 0
            and get_lone_pair(a1) > 0
            and has_empty_orbitals(a2)
        )

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of adjacent lone pair radical multiple bond delocalization.
        """
        a1_idx, a2_idx = path

        increment_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        increment_radical(mol.GetAtomWithIdx(a1_idx))
        decrement_radical(mol.GetAtomWithIdx(a2_idx))

        return mol


@PathFinderRegistry.register("reverse_adj_lone_pair_radical_multiple_bond")
class ReverseAdjacentLonePairRadicalMultipleBondPathFinder(PathFinder):
    """
    Find all the delocalization paths of atom1 which either can obtain a lonePair,
    has a radical, and is bonded by a double/triple bond. For example, [:N.]=[CH2] <=> [::N]-[.CH2]
    """

    template = Chem.MolFromSmarts(
        f"[{hetero_gain_lone_pair_lose_radical_lose_bond}:1]=,#[{atom_gain_radical_lose_bond}:2]"
    )

    reverse = "ForwardAdjacentLonePairRadicalMultipleBondPathFinder"
    reverse_abbr = "forward_adj_lone_pair_radical_multiple_bond"

    @staticmethod
    def verify(
        mol: Union["RWMol", "RDKitMol"],
        path: tuple,
    ) -> bool:
        """
        Verify that the adjacent lone pair radical multiple bond delocalization path is valid.

        Args:
            mol (RWMol or RDKitMol): The molecule to search.
            path (tuple): The adjacent lone pair multiple bond delocalization path to verify.

        Returns:
            bool: True if the adjacent lone pair multiple bond delocalization path is valid, False otherwise.
        """
        a1_idx, _ = path
        return mol.GetAtomWithIdx(a1_idx).GetNumRadicalElectrons() > 0

    @staticmethod
    @transform_pre_and_post_process
    def transform(
        mol: "Mol",
        path: tuple,
    ) -> Optional["Mol"]:
        """
        Transform resonance structures based on the provided path of adjacent lone pair radical multiple bond delocalization.
        """
        a1_idx, a2_idx = path

        decrement_order(mol.GetBondBetweenAtoms(a1_idx, a2_idx))
        decrement_radical(mol.GetAtomWithIdx(a1_idx))
        increment_radical(mol.GetAtomWithIdx(a2_idx))

        return mol


# def find_N5dc_radical_delocalization_paths(atom1)
# It is included in the original version
# But the relevant resonance transformation is recoverable
# by the above methods.
# Removed in the rewriting.
