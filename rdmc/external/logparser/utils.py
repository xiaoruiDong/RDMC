#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to read Gaussian output file.
"""

from collections import defaultdict
import re


# TODO: All information below need to be expanded
XTB = ['gfn0-xtb', 'gfn1-xtb', 'gfn2-xtb']
SEMIEMPIRICAL = ['am1', 'pm3', 'pm6'] + XTB
HF = ['hf']
MP_METHOD = ['mp2', 'mp3', 'mp4', 'mp4(dq)', 'mp4(sdq)', 'mp5']
DFT = ['b3lyp', 'cam-b3lyp', 'wb97xd', 'wb97x', 'apfd', 'b97d3', 'blyp', 'm06', 'm062x']
DOUBLE_HYBRID_DFT = ['b2plyp', 'mpw2plyp', 'b2plypd', 'b2plypd3', 'dsdpbep86',
                     'pbe0dh', 'pbeqidh', 'M08HX', 'MN15', 'MN15L']
COUPLE_CLUSTER = ['ccsd', 'ccsdt', 'ccsd(t)']
COMPOSITE_METHODS = ['cbs-qb3', 'rocbs-qb3', 'cbs-4m', 'cbs-apno', 'g1', 'g2', 'g3',
                     'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp2',
                     'w1u', 'w1bd', 'w1ro']
METHODS = SEMIEMPIRICAL + HF + DFT + MP_METHOD + DOUBLE_HYBRID_DFT + COUPLE_CLUSTER
DEFAULT_METHOD = 'hf'

STO = ['sto-3g', 'sto-6g', 'sto-3g*', 'sto-6g*', 'sto-6g**', 'sto-6g(d)', 'sto-6g(d,p)']
POPLE = ['3-21g', '4-31g', '6-21g', '6-31g', '6-311g',
         '3-21g*', '3-21g(d)', '3-21g**', '3-21g(d,p)', '3-21+g', '3-21+g*', '3-21+g(d)', '3-21+g**', '3-21+g(d,p)',
         '6-21g*', '6-21g(d)', '6-21g**', '6-21g(d,p)', '6-21+g', '6-21+g*', '6-21+g(d)', '6-21+g**', '6-21+g(d,p)',
         '6-31g*', '6-31g(d)', '6-31g**', '6-31g(d,p)', '6-31+g', '6-31+g*', '6-31+g(d)', '6-31+g**', '6-31+g(d,p)',
         '6-311g*', '6-311g(d)', '6-311g**', '6-311g(d,p)', '6-311+g', '6-311+g*', '6-311+g(d)', '6-311+g**', '6-311+g(d,p)',
         'mg3', 'mg3s', 'cbsb7', 'g3largemp2']
CORR_CS = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z', 'cc-pv6z',
           'aug-cc-pvdz', 'aug-cc-pvtz', 'aug-cc-pvqz', 'aug-cc-pv5z', 'aug-cc-pv6z',
           'jul-cc-pvdz', 'jul-cc-pvtz', 'jul-cc-pvqz', 'jul-cc-pv5z', 'jul-cc-pv6z',
           'jun-cc-pvdz', 'jun-cc-pvtz', 'jun-cc-pvqz', 'jun-cc-pv5z', 'jun-cc-pv6z',
           'cc-pcvdz', 'cc-pcvtz', 'cc-pcvqz', 'cc-pcv5z', 'cc-pcv6z',
           'aug-cc-pcvdz', 'aug-cc-pcvtz', 'aug-cc-pcvqz', 'aug-cc-pcv5z', 'aug-cc-pcv6z']
KARLSRUHE = ['def2sv', 'def2svp', 'def2svpp', 'def2svpd', 'def2svppd',
             'def2tzv', 'def2tzvp', 'def2tzvpp', 'def2tzvpd', 'def2tzvppd',
             'def2qzv', 'def2qzvp', 'def2qzvpp', 'def2qzvpd', 'def2qzvppd']
BASIS_SETS = STO + POPLE + CORR_CS + KARLSRUHE
DEFAULT_BASIS_SET = 'sto-3g'

SCHEME_REGEX = r'([\w\-]+=?\([^\(\)]+\)|\w+=\w+|[^\s=\(\)]+)'


def scheme_to_dict(scheme_str: str) -> dict:
    """
    A function to transform scheme to a dict.

    Args:
        scheme_str (str): the calculation scheme used in a Gaussian job.
    """
    # Remove the header of the line
    if scheme_str.startswith('#p') or scheme_str.startswith('#n'):
        scheme_str = scheme_str[2:]
    elif scheme_str.startswith('#'):
        scheme_str = scheme_str[1:]

    # External
    if 'external' in scheme_str:
        scheme_str, external_str = scheme_str.split('external', 1)
        external_dict = {'external': external_str.strip()[1:]}
    else:
        external_dict = {}

    # split other scheme arguments
    args = re.findall(SCHEME_REGEX, scheme_str)

    schemes = defaultdict(lambda:{})
    for arg in args:
        if arg.startswith('#'):
            continue
        elif 'iop' in arg:
            # Example: iop(7/33=1,2/16=3)
            iop_scheme = schemes['iop']
            content = arg.split('(')[1].split(')')[0]
            for item in content.split(','):
                # items may be split by ','
                try:
                    key, val = item.strip().split('=')
                except:
                    # There may be items without assigned values
                    key, val = item.strip(), None
                iop_scheme[key.strip()] = val.strip()
        elif '=' in arg and '(' in arg:
            scheme_name = arg.split('(')[0].replace('=', '').strip()
            scheme_dict = schemes[scheme_name]
            scheme_vals = arg.split('(')[1].split(')')[0].split(',')
            for val in scheme_vals:
                val = val.strip()
                if '=' not in val:
                    scheme_dict[val] = True
                else:
                    k, v = val.split('=')
                    scheme_dict[k.strip()] = v.strip()
        elif '=' in arg:
            key, val = arg.split('=')
            scheme_dict = schemes[key.strip()]
            scheme_dict[val.strip()] = True
        elif '//' in arg:
            scheme_dict = schemes['LOT']
            sp_lot, opt_lot = arg.split('//')
            scheme_dict['SP'] = sp_lot.strip()
            scheme_dict['OPT'] = opt_lot.strip()
            scheme_dict['LOT'] = arg.strip()
        elif '/' in arg:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in SEMIEMPIRICAL:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in COMPOSITE_METHODS:
            scheme_dict = schemes['LOT']
            scheme_dict['LOT'] = arg.strip()
        elif arg in METHODS:
            scheme_dict = schemes['LOT']
            scheme_dict['method'] = arg.strip()
        elif arg in BASIS_SETS:
            scheme_dict['basis_set'] = arg.strip()
        else:
            schemes[arg.strip()] = True

    schemes.update(external_dict)

    if not schemes.get('LOT') and schemes.get('external'):
        schemes['LOT'] = {'LOT': schemes['external']}
    elif not schemes.get('LOT'):
        schemes['LOT'] = {'LOT': f'{DEFAULT_METHOD}/{DEFAULT_BASIS_SET}'}
    elif schemes['LOT'].get('method') and not schemes['LOT'].get('basis_set'):
        schemes['LOT']['LOT'] = f"{schemes['LOT']['method']}/{DEFAULT_BASIS_SET}"
    elif not schemes['LOT'].get('method') and schemes['LOT'].get('basis_set'):
        schemes['LOT']['LOT'] = f"{DEFAULT_METHOD}/{schemes['LOT']['basis_set']}"
    if schemes.get('freq'):
        schemes['LOT']['freq'] = schemes['LOT']['LOT']

    return schemes
