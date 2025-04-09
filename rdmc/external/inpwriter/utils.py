#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Tuple


XTB_GAUSSIAN_PERL_PATH = Path(__file__).parent / 'xtb_gaussian.pl'


def _avoid_empty_line(str_block: str,
                      ) -> str:
    """
    A helper function to avoid introducing empty line in the input file.

    Args:
        str_block (str): The string block to be written.

    Return:
        str: The string block to be written.
    """
    return str_block.strip() + '\n' if str_block else ''


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
