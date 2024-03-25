from typing import Tuple

from rdtools.compare import get_match_and_recover_recipe, is_same_connectivity_mol, has_matched_mol


class MolCompareMixin:

    def GetMatchAndRecoverRecipe(
        self,
        mol: "Mol",
    ) -> Tuple[tuple, dict]:
        """
        Get the isomorphic match between two molecules and a recipe to recover
        the provide `mol` to the current mol. If swapping the atom indices in `mol` according to the recipe,
        the `mol` should have the same connectivity as the current molecule. Note, if no match is found,
        the returned match and recipe will be empty.

        Args:
            mol (RDKitMol): The molecule to compare with.

        Returns:
            tuple: The substructure match.
            dict: A truncated atom mapping of mol2 to mol1.
        """
        return get_match_and_recover_recipe(self, mol)

    def IsSameConnectivity(
        self,
        mol: "Mol",
    ) -> bool:
        """
        Check wheter the molecule has the same connectivity as the reference molecule.

        Args:
            mol (RDKitMol): The reference molecule.

        Returns:
            bool: Whether the molecule has the same connectivity as the reference molecule.
        """
        return is_same_connectivity_mol(self, mol)

    def HasMatchedMol(
        self,
        mols: list,
        considerAtomMapping: bool = False,
    ) -> bool:
        """
        Check whether the current molecule has a match to any of the provided molecules.

        Args:
            mols (list): The list of molecules to compare with.
            considerAtomMapping (bool, optional): Whether to consider the atom mapping of the provided molecules.

        Returns:
            bool: Whether the current molecule has the any of the provided molecules.
        """
        return has_matched_mol(self, mols, considerAtomMapping)
