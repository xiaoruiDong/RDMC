import numpy as np

from rdkit import Chem

from rdtools.bond import get_all_changing_bonds


def is_DA_rxn_endo(
    rmol: 'RDKitMol',
    pmol: 'RDKitMol',
    embed: bool = False
) -> bool:
    """
    Determine the Diels Alder reaction stereo type (endo or exo),
    based on the provided reactants and products. The two molecules must be atom
    mapped.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        embed (bool): bool. If the DA product has no conformer embedded.
                            Whether to embed a conformer. Defaults to ``False``.

    Returns:
        bool: if the reaction has an endo configuration
    """
    frags = Chem.GetMolFrags(rmol)

    if len(frags) == 1:
        # This reaction is defined in the reverse direction:
        # DA_product <=> diene + dienophile
        rmol, pmol = pmol, rmol
        frags = Chem.GetMolFrags(rmol)

    assert len(frags) == 2

    if not pmol.GetNumConformers():
        if embed:
            pmol.EmbedConformer()
        else:
            raise ValueError(
                'The provided DA product has no geometry available'
                'Cannot determine the stereotype of the DA reaction'
            )

    # Analyze the reaction center
    formed, broken, changing = get_all_changing_bonds(rmol, pmol)
    assert len(formed) == 2 and len(broken) == 0

    # `fbond_atoms` are atoms in the formed bonds
    fbond_atoms = set([atom for bond in formed for atom in bond])
    for bond in changing:
        bond = set(bond)
        if len(bond & fbond_atoms) == 0:
            # Find the single bond in the diene
            dien_sb = list(bond)
        elif len(bond & fbond_atoms) == 2:
            # Find the double bond of the dienophile
            dinp_db = list(bond)
    # Make `fbond_atoms` for convenience in slicing
    fbond_atoms = list(fbond_atoms)

    # Find the atom indexes in diene and dienophile
    _, dienophile = frags if dien_sb[0] in frags[0] else frags[::-1]

    # Get the 3D geometry of the DA product
    # Create a reference plane from atoms in formed bonds
    # The reference point is chosen to be the center of the plane
    xyz = pmol.GetConformer().GetPositions()
    ref_plane = xyz[fbond_atoms]
    ref_pt = ref_plane.mean(axis=0, keepdims=True)

    # Use the total least square algorithm to find
    # the normal vector of the reference plane
    A = ref_plane - ref_pt
    norm_vec = np.linalg.svd(A.T @ A)[0][:, -1].reshape(1, -1)

    # Use the vector between middle point of the diene single point
    # and the reference point as one direction vector
    dien_vec = xyz[dien_sb, :].mean(axis=0, keepdims=True) - ref_pt

    # Use the vector between mass center of the dienophile
    # and the reference point as one direction vector
    # excluding atom in dienophile's double bond
    atom_scope = [atom for atom in dienophile if atom not in dinp_db]
    mass_vec = [rmol.GetAtomWithIdx(i).GetMass() for i in atom_scope]
    wt_xyz = (xyz[atom_scope, :] * np.reshape(mass_vec, (-1, 1)))
    dinp_vec = wt_xyz.mean(axis=0, keepdims=True) - ref_pt

    # Endo is determined by if dien_vec has the same direction as the dinp_vec
    # using the normal vector of the reference plane as a reference direction
    endo = ((norm_vec @ dien_vec.T) * (norm_vec @ dinp_vec.T)).item() > 0

    return endo
