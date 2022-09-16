# standard library imports
import subprocess
import sys
import os
from os import path

# third party
import numpy as np
import cclib

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from .base_lot import Lot

from rdmc.external.xtb_tools.utils import XTB_GAUSSIAN_PL

class Gaussian(Lot):

    def __init__(self, options):
        super(Gaussian, self).__init__(options)

        gaussian_scratch = os.environ.get("GAUSS_SCRDIR")
        if not os.path.exists(gaussian_scratch):
            os.makedirs(gaussian_scratch)

    def write_input_file(self, geom, multiplicity):

        if self.lot_inp_file is None:
            inpstring = (
                f"%mem=1gb\n"
                f"%nprocshared={self.nproc}\n"
                f"#N NoSymmetry scf(xqc) force\n"
                f'external="{XTB_GAUSSIAN_PL} --gfn 2 -P\n"'
                f"\n"
                f"Title Card Required"
            )
        else:
            inpstring = ""
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                inpstring += line

        inpstring = (f"{inpstring}\n"
                     f"\n"
                     f"{self.charge} {multiplicity}\n"
        )
        for coord in geom:
            for i in coord:
                inpstring += str(i) + " "
            inpstring += "\n"
        inpstring += "\n"
        gaussian_scratch = os.environ.get("GAUSS_SCRDIR")
        tempfilename = os.path.join(gaussian_scratch, "gaussian_force.gjf")
        tempfile = open(tempfilename, "w")
        tempfile.write(inpstring)
        tempfile.close()
        return tempfilename

    def run(self, geom, multiplicity, ad_idx, runtype="gradient"):

        assert ad_idx == 0, "pyGSM Gaussian doesn't currently support ad_idx!=0"

        for version in ["g16", "g09", "g03"]:
            GAUSSIAN_ROOT = os.environ.get(f"{version}root")
            if GAUSSIAN_ROOT:
                break
        else:
            raise RuntimeError("No Gaussian installation found.")

        gaussian_binary = os.path.join(GAUSSIAN_ROOT, version, version)

        # Run the gaussian via subprocess
        gaussian_scratch = os.environ.get("GAUSS_SCRDIR")
        gaussian_input_file = self.write_input_file(geom, multiplicity)
        gaussian_output_file = os.path.join(gaussian_scratch, "gaussian_force.log")
        with open(gaussian_output_file, "w") as f:
            gaussian_run = subprocess.run(
                [gaussian_binary, gaussian_input_file],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

        # parse output
        self.parse(gaussian_output_file, multiplicity)

        return

    def parse(self, gaussian_output_file, multiplicity):
        p = cclib.parser.Gaussian(gaussian_output_file)
        data = p.parse()
        energies = data.scfenergies # eV unit
        grads = data.grads
        coords = data.atomcoords
        energy = energies[0] * 0.036749326681 # eV -> Hartree
        gradient = -grads[0] # Hartree/Bohr
        self._Energies[(multiplicity, 0)] = self.Energy(energy, "Hartree")
        self._Gradients[(multiplicity, 0)] = self.Gradient(gradient, "Hartree/Bohr")
