from rdkit import Chem

class MolTransformerMixin:

    @classmethod
    def FromSmarts(
        cls,
        smarts: str,
        **kwargs,
    ):
        """
        Convert a SMARTS to an ``RDKitMol`` object.

        Args:
            smarts (str): A SMARTS string of the molecule

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the SMARTS.
        """
        mol = Chem.MolFromSmarts(smarts, **kwargs)
        return cls(mol)