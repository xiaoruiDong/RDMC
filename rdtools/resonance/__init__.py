#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modules for resonance structure generation and analysis."""

# Make sure resonance algorithms are registered
import rdtools.resonance.rdkit_backend
import rdtools.resonance.rmg_backend
from rdtools.resonance.base import generate_resonance_structures
