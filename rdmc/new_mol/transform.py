from rdkit import Chem

from rdmc.rdtools.conversion.smiles import mol_from_smiles
from rdmc.rdtools.conversion.rmg import mol_from_rmg_mol
from rdmc.rdtools.conversion.xyz import mol_from_xyz
from rdmc.rdtools.obabel import (
    openbabel_mol_to_rdkit_mol as mol_from_openbabel_mol,
)

class MolTransformerMixin:

    @classmethod
    def FromSmarts(
        cls,
        smarts: str,
        **kwargs,
    ):
        """
        Convert a SMARTS to a new molecule object.

        Args:
            smarts (str): A SMARTS string of the molecule

        Returns:
            Mol: A molecule object corresponding to the SMARTS.
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
        Convert a InChI string to a new molecule object.

        Args:
            inchi (str): A InChI string. https://en.wikipedia.org/wiki/International_Chemical_Identifier
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Due to RDKit implementation,
                                       only effective when sanitize is ``True`` as well. ``True`` to remove.
            addHs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                    Only functioning when ``removeHs`` is ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            Mol: A molecule object corresponding to the InChI.
        """
        mol = Chem.inchi.MolFromInchi(inchi, sanitize=sanitize, removeHs=removeHs)
        if not removeHs and addHs:
            mol = Chem.AddHs(mol)
        return cls(mol)

    @classmethod
    def FromRMGMol(
        cls,
        rmgMol: "rmgpy.molecule.Molecule",
        removeHs: bool = False,
        sanitize: bool = True,
    ):
        """
        Convert an RMG ``Molecule`` to a new molecule object.

        Args:
            rmgMol ('rmg.molecule.Molecule'): An RMG ``Molecule`` instance.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            Mol: A molecule object corresponding to the RMG Molecule.
        """
        return cls(mol_from_rmg_mol(rmgMol, removeHs, sanitize))

    @classmethod
    def FromXYZ(
        cls,
        xyz: str,
        backend: str = "openbabel",
        header: bool = True,
        sanitize: bool = True,
        embed_chiral: bool = False,
        **kwargs,
    ):
        """
        Convert a xyz string to a new molecule object.

        Args:
            xyz (str): A XYZ String.
            backend (str): The backend used to perceive molecule. Defaults to ``'openbabel'``.
                            Currently, we only support ``'openbabel'`` and ``'xyz2mol'``.
            header (bool, optional): If lines of the number of atoms and title are included.
                                    Defaults to ``True.``
            sanitize (bool): Sanitize the RDKit molecule during conversion. Helpful to set it to ``False``
                            when reading in TSs. Defaults to ``True``.
            embed_chiral: ``True`` to embed chiral information. Defaults to ``True``.
            supported kwargs:
                xyz2mol:
                    - charge: The charge of the species. Defaults to ``0``.
                    - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to ``False``.
                    - use_graph: ``True`` to use networkx module for accelerate. Defaults to ``True``.
                    - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to ``False``.
                    - forced_rdmc: Defaults to ``False``. In rare case, we may hope to use a tailored
                                    version of the Jensen XYZ parser, other than the one available in RDKit.
                                    Set this argument to ``True`` to force use RDMC's implementation,
                                    which user's may have some flexibility to modify.

            Returns:
                Mol: A molecule object corresponding to the xyz.
        """
        return cls(mol_from_xyz(xyz, backend, header, sanitize, embed_chiral, **kwargs))

    @classmethod
    def FromMolBlock(
        cls,
        molBlock: str,
        removeHs: bool = False,
        sanitize: bool = True,
    ):
        """
        Convert a string containing the Mol block to a molecule object.

        Args:
            MolBlock (str): string containing the Mol block.
            removeHs (bool): Whether or not to remove hydrogens from the input. Defaults to ``False``.
            sanitize (bool): Whether or not to use RDKit's sanitization algorithm to clean input; helpful to set this
                             to ``False`` when reading TS files. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the Mol block string.
        """
        return cls(Chem.MolFromMolBlock(molBlock, removeHs=removeHs, sanitize=sanitize))

    @classmethod
    def FromOBMol(
        cls,
        obMol: "openbabel.OBMol",
        removeHs: bool = False,
        sanitize: bool = True,
        embed: bool = True,
    ):
        """
        Convert a OpenBabel Mol to a molecule object.

        Args:
            obMol (Molecule): An OpenBabel Molecule object for the conversion.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to ``True``.
            embed (bool, optional): Whether to embed 3D conformer from OBMol. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the input OpenBabel Molecule object.
        """
        return cls(mol_from_openbabel_mol(obMol, removeHs, sanitize, embed))

    @classmethod
    def FromSmiles(
        cls,
        smiles: str,
        remove_hs: bool = False,
        add_hs: bool = True,
        sanitize: bool = True,
        allow_cxsmiles: bool = True,
        keep_atom_map: bool = True,
        assign_atom_map: bool = True,
    ):
        """
        Convert a SMILES string to a molecule object.

        Args:
            smiles (str): A SMILES representation of the molecule.
            remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            add_hs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                    Only functioning when removeHs is False.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.
            allow_cxsmiles (bool, optional): Whether to recognize and parse CXSMILES. Defaults to ``True``.
            keep_atom_map (bool, optional): Whether to keep the atom mapping contained in the SMILES. Defaults
                                            Defaults to ``True``.
            assign_atom_map (bool, optional): Whether to assign the atom mapping according to the atom index
                                              if no atom mapping available in the SMILES. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the SMILES.
        """
        return cls(mol_from_smiles(smiles, remove_hs, add_hs, sanitize, allow_cxsmiles, keep_atom_map, assign_atom_map))
    
