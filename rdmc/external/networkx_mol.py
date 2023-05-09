#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides methods that utilizing networkx with RDKitMol.
"""

from copy import deepcopy

import networkx as nx
from networkx.algorithms.isomorphism import ISMAGS
from rdmc.utils import CPK_COLOR_PALETTE


def to_graph(mol: 'RDKitMol',
             keep_bond_order=False):
    """
    Convert a RDKitMol to a networkx graph.
    """
    nx_graph = nx.Graph()

    for atom in mol.GetAtoms():
        nx_graph.add_node(atom.GetIdx(),
                          symbol=atom.GetSymbol(),
                          atomic_num=atom.GetAtomicNum(),
                          node_color=CPK_COLOR_PALETTE[atom.GetSymbol()],
                          )

    for bond in mol.GetBonds():
        bond_type = 1 if not keep_bond_order else bond.GetBondTypeAsDouble()
        nx_graph.add_edge(bond.GetBeginAtomIdx(),
                          bond.GetEndAtomIdx(),
                          bond_type=bond_type,
                          )

    return nx_graph


def draw_networkx_mol(molgraph: nx.Graph) -> None:
    """
    Draw a networkx graph.

    Args:
        molgraph (nx.Graph): A networkx graph representing a molecule.
    """
    # labels as element:atom_index
    labels = {i: f'{symbol}:{i}'
              for i, symbol in nx.get_node_attributes(molgraph,
                                                      name="symbol",
                                                      ).items()}
    # node color should be input as a list of colors
    node_colors = list(nx.get_node_attributes(molgraph, 'node_color').values())
    nx.draw(molgraph,
            with_labels=True,
            node_size=1000,
            labels=labels,
            node_color=node_colors,
            edgecolors='black',)

