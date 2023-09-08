RDMC Documentation
===================

**RDMC (Reaction Data and Molecule Conformer)** is A light-weight wrapper for **RDKit** Molecule and Conformer related operations. RDkit is great! I like its versatility and speed, but
personally, I find learning to use RDKit is not easy. I usually have to go back and forth to check if RDKit has certain methods and how to use them,
since those molecule operations are usually located in different modules. I wrote this tiny thing majorly aiming to make my life easier and to
provide a convenient tool so that users can just import a single module / class instead of remembering which method in what module.

To start with, simply try::

    from rdmc.mol import RDKitMol

and see what you can do with this ``RDKitMol`` class!


Contents
========
.. toctree::
   :maxdepth: 2

   reference/rdmc
   license


Indices and tables
===================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
