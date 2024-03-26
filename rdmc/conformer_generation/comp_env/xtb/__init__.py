#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.conformer_generation.comp_env.software import register_binary, try_import

register_binary("xtb")
register_binary("crest")

try_import("xtb.ase.calculator.XTB", "xtb_calculator", globals(), "xtb-python")
