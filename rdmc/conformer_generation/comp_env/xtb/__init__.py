#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.comp_env.software import has_binary, try_import


xtb_available = has_binary("xtb")
crest_available = has_binary("crest")

package_name = "xtb-python"
namespace = globals()
modules = [
    ("xtb.ase.calculator.XTB", "xtb_calculator"),
    "xtb.libxtb.VERBOSITY_FULL",
    "xtb.libxtb.VERBOSITY_MINIMAL",
    "xtb.libxtb.VERBOSITY_MUTED",
    "xtb.utils.get_method",
    "xtb.utils._methods",
    "xtb.interface.Calculator",
]

for module in modules:
    if isinstance(module, tuple):
        try_import(
            module[0], alias=module[1], namespace=namespace, package_name=package_name
        )
    else:
        try_import(module, namespace=namespace, package_name=package_name)
