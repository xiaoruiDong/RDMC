#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.comp_env.software import has_binary, try_import


xtb_available = has_binary("xtb")
crest_available = has_binary("crest")

try_import("xtb.ase.calculator.XTB", "xtb_calculator", globals(), "xtb-python")
