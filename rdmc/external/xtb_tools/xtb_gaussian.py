#!/.conda/envs/rdmc/bin/python

"""
XTB + gaussian optimization. Modified based on https://github.com/jensengroup/ReactionDiscovery/blob/main/xtb_ts_test/xtb_external.py
This can be used as an external script for Gaussian. In the shebang, make sure to use the python binary that has the correct environment.
Also, change the file's mode to 755, so that Gaussian can call this file. SET XTB_BINARY to the xtb's binary.
"""

import os
import sys
import subprocess

import numpy as np
import fortranformat as ff
# Extra dependency
# Use conda install -c conda-forge fortranformat or pip install fortranformat

# Hard coded to avoid over head
XTB_BINARY = "/home/gridsan/groups/RMG/Software/xtb-6.4.1/bin/xtb"

XTB_ENV = {
    "OMP_STACKSIZE": "1G",
    "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "MKL_NUM_THREADS": "1",
}

# Avoid the usage of Chem periodic table
ELEM_TO_ATOMNUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

ATOMNUM_TO_ELEM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

def parse_ifile(ifile):
    """
    parse the ifile from xTB.
    """
    with open(ifile, "r") as ifile:
        gau_input = ifile.readlines()
        # tokens = ifile.readline()

    tokens = gau_input[0].split()
    natoms, nderiv, chrg, spin = [int(tokens[i]) for i in range(4)]

    coords = np.empty((natoms, 3))
    atomtypes = []
    for i, line in enumerate(gau_input[1:1 + natoms]):
        line = line.split()
        atomtypes.append(ATOMNUM_TO_ELEM.get(int(line[0])))
        coords[i] = np.array(list(map(float, line[1 : 1 + 3]))) * 0.529177249  # bohr to angstrom
    return natoms, nderiv, chrg, spin, atomtypes, coords


def parse_ofile(ofile, energy, natoms, dipole, gradient=None, hessian=None):
    """
    Parse the outfile for Gaussian to read.
    """
    headformat = ff.FortranRecordWriter("4D20.12")
    bodyformat = ff.FortranRecordWriter("3D20.12")

    with open(ofile, "w") as f:
        head = [energy, dipole[0], dipole[1], dipole[2]]
        headstring = headformat.write(head)
        f.write(headstring + "\n")

        if gradient is None:
            gradient = np.zeros((natoms, 3))

        for i in range(natoms):
            output = bodyformat.write(gradient[i])
            f.write(output + "\n")

        # polarizability and dipole derivatives are set to zero
        polarizability = np.zeros((2, 3))
        dipole_derivative = np.zeros((3 * natoms, 3))

        for i in range(2):
            output = bodyformat.write(polarizability[i])
            f.write(output + "\n")

        for i in range(3 * natoms):
            output = bodyformat.write(dipole_derivative[i])
            f.write(output + "\n")

        if hessian is not None:  # Only needed if ndreiv = 1
            tril = np.tril_indices(hessian.shape[0])
            tril_hessian = hessian[tril]
            for window in tril_hessian.reshape(int(tril_hessian.shape[0] / 3), 3):
                output = bodyformat.write(window)
                f.write(output + "\n")


def write_xyz(natoms, atomtypes, coords):
    """ Write .xyz file """
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atomtypes, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"

    with open("mol.xyz", "w") as inp:
        inp.write(xyz)


def get_energy(output):
    """
    Get total energy from the output.
    """
    for line in output.split("\n"):
        if "TOTAL ENERGY" in line:
            return float(line.split()[3])


def get_dipole(output):
    """
    Get dipole from the output.
    """
    tmp_output = output.split("molecular dipole:")
    del tmp_output[0]

    dipole_data = tmp_output[0].split("\n")
    dipole_line = dipole_data[3].split()

    dipole = np.array([dipole_line[1], dipole_line[2],
                       dipole_line[3]], dtype=float)
    return dipole


def get_gradient(natoms):
    """ """
    with open("gradient", "r") as grad_file:  # Not sure if a more specific path is needed
        gradient_data = grad_file.readlines()

    del gradient_data[: 2 + natoms]

    gradient = np.empty((natoms, 3))
    for i, line in enumerate(gradient_data[:natoms]):
        line = line.split()
        line = [num.replace("D", "E") for num in line]  # Original authors's comment: remove for new xtb version
        gradient[i, :] = line
    return gradient


def get_hessian(natoms):
    """ """
    with open("hessian", "r") as hess_file:
        hessian_data = hess_file.readlines()

    hessian = np.empty(3 * natoms * 3 * natoms)
    i = 0
    for line in hessian_data:
        if "$hessian" in line:
            continue

        for elm in line.strip().split():
            hessian[i] = float(elm)
            i += 1

    return hessian.reshape((3 * natoms, 3 * natoms))


def run_xtb(natoms, nderiv, chrg, spin, atomtypes, coords, solvent=None):
    """ """
    write_xyz(natoms, atomtypes, coords)

    cmd = f"{XTB_BINARY} mol.xyz --chrg {chrg} --uhf {spin - 1} --gfn 2 --parallel "
    if nderiv == 1:
        cmd += "--grad "
    elif nderiv == 2:
        cmd += "--hess --grad "

    if solvent is not None:
        method, solvent = solvent.split('=')
        cmd += f"--{method} {solvent} "

    print(cmd)
    p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=XTB_ENV,
        )

    output, _ = p.communicate()
    output = output.decode('utf-8')

    energy = get_energy(output)
    dipole = get_dipole(output)

    return energy, dipole


def clean_dir():
    """ delete all files """
    files = [
        "energy",
        "charges",
        "mol.xyz",
        "xtbrestart",
        "gradient",
        "hessian",
        "vibspectrum",
        "wbo",
        "mol.engrad",
        "xtbtopo.mol",
        "g98.out",
        "xtbhess.xyz",
    ]
    for _file in files:
        if os.path.exists(_file):
            os.remove(_file)


if __name__ == "__main__":

    solvent = None

    if len(sys.argv[1]) > 7: # given ekstra kwd
        for i, kwd in enumerate(sys.argv):
            if kwd == "R":
                break

            if "gbsa" or "alpb" in kwd:
                solvent = kwd

        ifile=sys.argv[i + 1]
        ofile=sys.argv[i + 2]

    else:
        ifile = sys.argv[2]
        ofile = sys.argv[3]

    (natoms, nderiv, chrg, spin, atomtypes, coords) = parse_ifile(ifile)

    energy, dipole = run_xtb(natoms, nderiv, chrg, spin, atomtypes, coords, solvent=solvent)

    if nderiv == 0:
        parse_ofile(ofile, energy, natoms, dipole)
    elif nderiv == 1:
        gradient = get_gradient(natoms)
        parse_ofile(ofile, energy, natoms, dipole, gradient=gradient)
    elif nderiv == 2:
        gradient = get_gradient(natoms)
        hessian = get_hessian(natoms)
        parse_ofile(ofile, energy, natoms, dipole, gradient=gradient, hessian=hessian)

    clean_dir()

