#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple


def _get_mult_and_chrg(mol: 'RDKitMol',
                       multiplicity: Optional[int] = None,
                       charge: Optional[int] = None,
                       ) -> Tuple[int, int]:
    """
    Get the multiplicity and charge of a molecule.

    Args:
        mol: The molecule.
        multiplicity: The multiplicity.
        charge: The charge.

    Returns:
        The multiplicity and charge of the molecule.
    """
    if multiplicity is None:
        multiplicity = mol.GetSpinMultiplicity()
    if charge is None:
        charge = mol.GetFormalCharge()
    return multiplicity, charge
