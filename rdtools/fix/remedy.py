"""
This module includes all the remedies for fixing molecules.
"""

from functools import lru_cache
from rdkit.Chem import rdChemReactions


@lru_cache(maxsize=1)
def get_recommend_remedies():
    return [
        # Remedy 1 - Carbon monoxide: [C]=O to [C-]#[O+]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v2X1:1]=[C+0-0v2X1:2]>>[O+1v3X1:1]#[C-1v3X1:2]"
        ),
        # Remedy 2 - Carbon monoxide: [C]=O to [C-]#[O+]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X1:1]#[C+0-0v3X1:2]>>[O+1v3X1:1]#[C-1v3X1:2]"
        ),
        # Remedy 3 - Oxygen Molecule: O=O to [O]-[O]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v2X1:1]=[O+0-0v2X1:2]>>[O+0-0v1X1:1]-[O+0-0v1X1:2]"
        ),
        # Remedy 4 - isocyanide: R[N]#[C] to R[N+]#[C-]
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X2:1]#[C+0-0v3X1:2]>>[N+v4X2:1]#[C-v3X1:2]"
        ),
        # Remedy 5 - azide: RN=N=[N] to RN=[N+]=[N-]
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v3X2:1]=[N+0-0v4X2:2]=[N+0-0v2X1:3]>>[N+0-0v3X2:1]=[N+1v4X2:2]=[N-1v2X1:3]"
        ),
        # Remedy 6 - amine oxide: RN(R)(R)-O to R[N+](R)(R)-[O-]
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X4:1]-[O+0-0v1X1:2]>>[N+1v4X4:1]-[O-1v1X1:2]"
        ),
        # Remedy 7 - amine radical: R[C](R)-N(R)(R)R to R[C-](R)-[N+](R)(R)R
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X4:1]-[C+0-0v3X3:2]>>[N+1v4X4:1]-[C-1v3X3:2]"
        ),
        # Remedy 8 - amine radical: RN(R)=C to RN(R)-[C]
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X3:1]=[C+0-0v4X3:2]>>[N+0-0v3X3:1]-[C+0-0v3X3:2]"
        ),
        # Remedy 9 - quintuple C bond, usually due to RC(=O)=O: R=C(R)=O to R=C(R)-[O]
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v5X3:1]=[O+0-0v2X1:2]>>[C+0-0v4X3:1]-[O+0-0v1X1:2]"
        ),
        # Remedy 10 - sulphuric bi-radicals: R[S](R)(-[O])-[O] to R[S](R)(=O)(=O)
        rdChemReactions.ReactionFromSmarts(
            "[S+0-0v4X4:1](-[O+0-0v1X1:2])-[O+0-0v1X1:3]>>[S+0-0v6X4:1](=[O+0-0v2X1:2])=[O+0-0v2X1:3]"
        ),
        # Remedy 11 - Triazinane: C1=N=C=N=C=N=1 to c1ncncn1
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v5X3:1]1=[N+0-0v4X2:2]=[C+0-0v5X3:3]=[N+0-0v4X2:4]=[C+0-0v5X3:5]=[N+0-0v4X2:6]=1"
            ">>[C+0-0v5X3:1]1[N+0-0v4X2:2]=[C+0-0v5X3:3][N+0-0v4X2:4]=[C+0-0v5X3:5][N+0-0v4X2:6]=1"
        ),
    ]


@lru_cache(maxsize=1)
def get_zwitterion_remedies():
    return [
        # Remedy 1 - criegee Intermediate: R[C](R)O[O] to RC=(R)[O+][O-]
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v3X3:1]-[O+0-0v2X2:2]-[O+0-0v1X1:3]>>[C+0-0v4X3:1]=[O+1v3X2:2]-[O-1v1X1:3]"
        ),
        # Remedy 2 - criegee Intermediate: [C]-C=C(R)O[O] to C=C-C=(R)[O+][O-]
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v3X3:1]-[C:2]=[C+0-0v4X3:3]-[O+0-0v2X2:4]-[O+0-0v1X1:5]>>[C+0-0v4X3:1]=[C:2]-[C+0-0v4X3:3]=[O+1v3X2:4]-[O-1v1X1:5]"
        ),
        # Remedy 3 - criegee like molecule: RN(R)(R)-C(R)(R)=O to R[N+](R)(R)-[C](R)(R)-[O-]
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X4:1]-[C+0-0v4X3:2]=[O+0-0v2X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
        ),
        # Remedy 4 - criegee like molecule: R[N+](R)(R)-[C-](R)(R)[O] to R[N+](R)(R)-[C](R)(R)-[O-]
        rdChemReactions.ReactionFromSmarts(
            "[N+1v4X4:1]-[C-1v3X3:2]-[O+0-0v1X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
        ),
        # Remedy 5 - ammonium + carboxylic: ([N]R4.C(=O)[O]) to ([N+]R4.C(=O)[O-])
        rdChemReactions.ReactionFromSmarts(
            "([N+0-0v4X4:1].[O+0-0v2X1:2]=[C+0-0v4X3:3]-[O+0-0v1X1:4])>>([N+1v4X4:1].[O+0-0v2X1:2]=[C+0-0v4X3:3]-[O-1v1X1:4])"
        ),
        # Remedy 6 - ammonium + phosphoric: ([N]R4.P(=O)[O]) to ([N+]R4.P(=O)[O-])
        rdChemReactions.ReactionFromSmarts(
            "([N+0-0v4X4:1].[P+0-0v5X4:2]-[O+0-0v1X1:3])>>([N+1v4X4:1].[P+0-0v5X4:2]-[O-1v1X1:3])"
        ),
        # Remedy 7 - ammonium + sulphuric: ([N]R4.S(=O)(=O)[O]) to ([N+]R4.S(=O)(=O)[O-])
        rdChemReactions.ReactionFromSmarts(
            "([N+0-0v4X4:1].[S+0-0v6X4:2]-[O+0-0v1X1:3])>>([N+1v4X4:1].[S+0-0v6X4:2]-[O-1v1X1:3])"
        ),
        # Remedy 8 - ammonium + carbonyl in ring: ([N]R4.C=O) to ([N+]R4.[C.]-[O-])
        rdChemReactions.ReactionFromSmarts(
            "([N+0-0v4X4:1].[C+0-0v4X3R:2]=[O+0-0v2X1:3])>>([N+1v4X4:1].[C+0-0v3X3R:2]-[O-1v1X1:3])"
        ),
    ]


@lru_cache(maxsize=1)
def get_ring_remedies():
    return [
        # The first four elements' sequence matters
        # TODO: Find a better solution to avoid the impact of sequence
        # Remedy 1 - quintuple C in ring: R1=C(R)=N-R1 to R1=C(R)[N]-R1
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v5X3R:1]=[N+0-0v3X2R:2]>>[C+0-0v4X3R:1]-[N+0-0v2X2R:2]"
        ),
        # Remedy 2 - quadruple N in ring: R1=N=C(R)R1 to R1=N-[C](R)R1
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v4X2R:1]=[C+0-0v4X3R:2]>>[N+0-0v3X2R:1]-[C+0-0v3X3R:2]"
        ),
        # Remedy 3 - ring =C(R)=N-[C]: R1=C(R)=N-[C](R)R1 to R1=C(R)-N=C(R)R1
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v5X3R:1]=[N+0-0v3X2R:2]-[C+0-0v3X3:3]>>[C+0-0v4X3R:1]-[N+0-0v3X2R:2]=[C+0-0v4X3:3]"
        ),
        # Remedy 4 - ring -N-N-: R1-N-N-R1 to R1-N=N-R1
        rdChemReactions.ReactionFromSmarts(
            "[N+0-0v2X2R:1]-[N+0-0v2X2R:2]>>[N+0-0v3X2R:1]=[N+0-0v3X2R:2]"
        ),
        # Remedy 5 - bicyclic radical
        rdChemReactions.ReactionFromSmarts(
            "[C+0-0v4:1]1[C+0-0v4X4:2]23[C+0-0v4:3][N+0-0v4X4:4]12[C+0-0v4:5]3>>[C+0-0v4:1]1[C+0-0v3X3:2]2[C+0-0v4:3][N+0-0v3X3:4]1[C+0-0v4:5]2"
        ),
    ]


@lru_cache(maxsize=1)
# This remedy is only used for oxonium species
def get_oxonium_remedies():
    return [
        # Remedy 1 - R[O](R)[O] to R[O+](R)[O-]
        # This is a case combining two radicals R-O-[O] and [R]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X3:1]-[O+0-0v1X1:2]>>[O+1v3X3:1]-[O-1v1X1:2]"
        ),
        # Remedy 2 - R[O](R)C(R)=O to R[O+](R)[C](R)[O-]
        # This is a case combining a closed shell ROR with a radical R[C]=O
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X3:1]-[C+0-0v4X3:2]=[O+0-0v2X1:3]>>[O+1v3X3:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
        ),
        # Remedy 3 - R[O](R)[C](R)R to R[O+](R)[C-](R)R
        # This is a case combining a radical R[C](R)(R) with a radical R[O]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X3:1]-[C+0-0v3X3:2]>>[O+1v3X3:1]-[C-1v3X3:2]"
        ),
    ]


class RemedyManager:
    """
    A class to manage all the remedies.
    """

    def __init__(self):
        self.remedies = {
            "recommend": get_recommend_remedies,
            "zwitterion": get_zwitterion_remedies,
            "ring": get_ring_remedies,
            "oxonium": get_oxonium_remedies,
        }

    def get_remedies(self, remedy_type: str):
        """
        Get the remedies of a specific type.

        Args:
            remedy_type (str): The type of remedies to be returned. Currently, only support ``recommend``, ``zwitterion``, ``ring`` and ``oxonium``.

        Returns:
            list: The remedies of the given type.
        """
        return self.remedies[remedy_type]() if remedy_type in self.remedies else []

    @property
    def default_remedies(self):
        """
        Get the default remedies.

        Returns:
            list: The default remedies.
        """
        return self.get_remedies("recommend")

    @property
    def all_remedies(self):
        """
        Get all the remedies.

        Returns:
            list: All the remedies.
        """
        return sum(
            [val() for key, val in self.remedies.items() if key != "oxonium"], []
        )


remedy_manager = RemedyManager()
