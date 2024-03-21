#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import RDLogger


# Mute RDKit's error logs
# They can be confusing at places where try ... except ... are implemented.
RDLogger.DisableLog("rdApp.*")