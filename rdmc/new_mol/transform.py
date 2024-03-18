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

    @classmethod
    def FromInchi(
        cls,
        inchi: str,
        removeHs: bool = False,
        addHs: bool = True,
        sanitize: bool = True,
    ):
        """
        Construct an ``RDKitMol`` object from a InChI string.

        Args:
            inchi (str): A InChI string. https://en.wikipedia.org/wiki/International_Chemical_Identifier
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Due to RDKit implementation,
                                       only effective when sanitize is ``True`` as well. ``True`` to remove.
            addHs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                    Only functioning when ``removeHs`` is ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            RDKitMol: An RDKit molecule object corresponding to the InChI.
        """
        mol = Chem.inchi.MolFromInchi(inchi, sanitize=sanitize, removeHs=removeHs)
        if not removeHs and addHs:
            mol = Chem.AddHs(mol)
        return cls(mol)
