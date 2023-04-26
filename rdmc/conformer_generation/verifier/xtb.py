#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from typing import Optional

from rdmc.conformer_generation.verifier.freq import FreqVerifier
from rdmc.conformer_generation.utils import _software_available
from rdmc.external.xtb_tools.run_xtb import run_xtb_calc
from rdmc.external.xtb_tools.utils import XTB_BINARY

_software_available['xtb'] = osp.isfile(XTB_BINARY)


class XTBFreqVerifier(FreqVerifier):
    """
    The class for verifying the species or TS by calculating and checking its frequencies using XTB.
    Since frequency may be calculated in an previous job. The class will first check if frequency
    results are available. If not, it will launch jobs to calculate frequencies.

    Args:
        method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
        cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                                     imaginary frequency to avoid small magnitude frequencies which correspond
                                     to internal bond rotations. This is only used when the molecule is a TS.
                                     Defaults to -100 cm^-1.
    """

    request_external_software = ['xtb']

    def task_prep(self,
                  method: str = "gfn2",
                  cutoff_freq: float = -100.,
                  **kwargs,
                  ):
        """
        Set the method and cutoff.

        Args:
            method (str, optional): The method used to calculate frequencies. Defaults to "gfn2".
            cutoff_freq (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
        """
        super().task_prep(cutoff_freq=cutoff_freq)
        self.method = method

    def calc_freq(self,
                  mol: 'RDKitMol',
                  multiplicity: Optional[int] = None,
                  charge: Optional[int] = None,
                  **kwargs):
        """
        Calculate the frequencies using xTB.

        Args:
            mol (RDKitMol)
            multiplicity (int): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
            charge (int): The charge of the molecule. Defaults to None, to use the charge of mol.
        """
        mult, charge = self._get_mult_and_chrg(mol, multiplicity, charge)

        run_ids = getattr(mol, 'keep_ids', [True] * mol.GetNumConformers())
        for cid, keep_id in enumerate(run_ids):
            if not keep_id:
                continue
            try:
                _, props = run_xtb_calc(mol,
                                        conf_id=cid,
                                        charge=charge,
                                        uhf=mult - 1,
                                        job="--hess",
                                        method=self.method)
                mol.frequencies[cid] = props["frequencies"]
            except Exception as exc:
                mol.frequencies[cid] = None
                mol.keep_ids[cid] = False
                print(exc)
