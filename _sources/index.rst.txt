RDMC Documentation
===================

.. image:: _static/RDMC_icon.svg
   :align: center

Introduction
-------------

**RDMC (Reaction Data and Molecular Conformer)** is an open-source lightweight software package specialized in handling Reaction Data and Molecular (including transition states) Conformers.

It contains various modules and classes helpful for relevant tasks to make conversion, visualization, manipulation, and analysis of molecules easier. ``rdmc`` and ``rdtools`` are the two major modules.


Installation
-------------
The source code of the RDMC software package is hosted on GitHub, and its binary distribution is available on Anaconda Cloud and PyPI. The easiest way to install RDMC is to use ``conda`` or ``mamba``::

    conda install -c xiaoruidong rdmc  # replace conda to mamba to use mamba

Or

.. code-block:: bash

    pip install rdmc

``conda`` can be installed by via `Anaconda <https://www.anaconda.com/download/>`_, and ``mamba`` can be installed via `Mambaforge <https://github.com/conda-forge/miniforge>`_.

You can also install RDMC from the source code:

.. code-block:: bash

    git clone https://github.com/xiaoruidong/rdmc
    cd RDMC
    conda env create -f environment.yml
    conda activate rdmc
    python -m pip install --no-deps -vv ./


``rdtools``
-----------
It has a collections of functions that can be directly operated on to RDKit native objects. Most of them can be regarded as a modified version of a RDKit native operation but with simplified
imports, more intuitive usage, and more robustness implementation. They are distributed in submodules, each named by its relevance field and objects. It is best used in cases where efficiency is a
major conern and the task you required is relatively simple. Here are some highlights in ``rdtools``:

* viewers in ``rdtools.view``. These viewers greatly extend RDKit's default Ipython 3D viewer, with the capability of viewing animations and interaction with conformers.
* ``generate_resonance_structures`` in ``rdtools.resonance`` is able to generate resonance structures for radical molecules that is not capable by the original RDKit.
* ``mol_from_smiles`` in ``rdtools.conversion`` make sure the created molecule has an atom ordering consistent with the atom mapping
* ``mol_from_xyz`` supports two backends ``openbabel`` as well as ``xyz2mol`` for molecule connectivity perception. If both native backends fail (e.g., cannot be sanitized, or wrong charge or multiplicity), ``rdtools`` also provided a heuristic fix tool ``fix_mol`` in ``rdtools.fix`` to help fix the molecules if possible.


``rdmc``
----------
It can be regarded as a midware between RDKit/``rdtools`` basic operations and complicated workflows. ``Mol`` (Previously, ``RDKitMol``) and ``Reaction`` are the most important classes.

* ``Mol`` (known as ``RDKitMol`` previously) is a child class of ``RWMol``, which means that you can directly use it with RDKit's native functions, but also integrated a lot of tools in ``rdtools``, so you can directly use them as class methods. The appended methods not only provides convenience in usage, but also make sure the output molecule objects, if applicable, is still a ``rdmc.Mol`` object. While many RDKit functions will just output ``Chem.Mol`` which is no longer editable, breaking the flow of your molecule operations.
* ``Reaction`` provides intuitive APIs for getting bond analysis, reaction comparison, visualization, etc.


We provide examples of how to combine ``rdtools`` and ``rdmc`` with other dependencies to build useful tools. Python native interactive Log file parsers in ``rdmc.external.logparse`` are a good show case
where ``rdmc`` and ``cclib`` are combined. We also provide solutions to pipelining tasks to achieve high-throughput generating and processing of large amount of molecule/reaction data in ``rdmc.conformer_generation``.

Requirements
------------
``RDMC`` is written in Python (>= 3.7) and has dependencies only on popular packages.

* To use ``rdtools``, you only needs ``numpy`` and ``rdkit`` at minimum. You can install optional dependencies: ``scipy`` for better resonance structure generation for polycyclic molecules, ``py3dmol`` to use the amazing 3D viewers, ``openbabel`` to extend `rdmc`'s xyz perception cability.
* To use ``rdmc``, the dependencies are basically the same as ``rdtools``, but we do recommend installing all optional dependencies for a better experience. Besides, to plot curves and figures for data, you can install ``matplotlib``; to play around with the log parsers you should consider install ``cclib`` and ``ipywidgets``. And to start computations in ``conformer_generation``, you need to have ``xtb`` and ``orca`` (which are free to academia) installed to get some serious results.

But in a word, RDMC's dependencies are very general-purpose and popular packages.


First Run
----------
To start with, simply try the following:

.. code-block:: python

    from rdmc import Mol, Reaction
    mol = Mol.FromSmiles('CCO')
    rxn = Reaction.from_reaction_smiles('CCO>>CC(=O)O')

And see what the ``mol`` and ``rxn`` are capable of! The full lists of APIs of :obj:`Mol <rdmc.mol.RDKitMol>` and :obj:`Reaction <rdmc.reaction.Reaction>` are provided in this documentation.


Examples
--------
We also provided a few notebooks (available in ``\ipython`` and `Colab <https://drive.google.com/drive/folders/1bHSChJOocycE5ZfbnGkocX4-OFVDhShW?usp=sharing>`_) to demonstrate the usage of RDMC. Please feel invited to try them out!


Developers
-----------
- Dr. Xiaorui Dong (|github_xiaorui|_ \| |linkedin_xiaorui|_ \| |gs_xiaorui|_),
- Dr. Lagnajit Pattanaik (|github_lucky|_ \| |linkedin_lucky|_ \| |gs_lucky|_),
- Dr. Shih-Cheng Li (|github_shihcheng|_ \| |linkedin_shihcheng|_ \| |gs_shihcheng|_),
- Dr. Kevin Spiekermann (|github_kevin|_ \| |linkedin_kevin|_ \| |gs_kevin|_),
- Dr. Hao-Wei Pang (|github_haowei|_ \| |linkedin_haowei|_ \| |gs_haowei|_),
- Jonathan W. Zheng (|github_jwz|_ \| |linkedin_jwz|_ \| |gs_jwz|_),
- Prof. William H. Green (|linkedin_bill|_ \| |gs_bill|_)

at `Green Research Group <https://greengroup.mit.edu>`_ at `Massachusetts Institute of Technology (MIT) <https://www.mit.edu>`_.
For any questions while using RDMC, please contact us via the `GitHub issue page <https://github.com/xiaoruiDong/RDMC/issues>`_ or email us at `rdmc_dev@mit.edu <mailto:rdmc@mit.edu>`_.


Contents
========
.. toctree::
   :maxdepth: 2

   reference/rdmc/rdmc
   reference/rdtools/rdtools
   reference/credits
   reference/cite
   reference/license


APIs
===================
* :ref:`genindex`
* :ref:`modindex`


.. |github| image:: _static/GitHub_icon.svg
    :class: social-icon
.. |github_xiaorui| replace:: |github|
.. _github_xiaorui: https://github.com/xiaoruiDong
.. |github_lucky| replace:: |github|
.. _github_lucky: https://github.com/PattanaikL
.. |github_shihcheng| replace:: |github|
.. _github_shihcheng: https://github.com/shihchengli
.. |github_kevin| replace:: |github|
.. _github_kevin: https://github.com/kspieks
.. |github_haowei| replace:: |github|
.. _github_haowei: https://github.com/hwpang
.. |github_jwz| replace:: |github|
.. _github_jwz: https://github.com/jonwzheng


.. |linkedin| image:: _static/LinkedIn_icon.svg
    :class: social-icon
.. |linkedin_xiaorui| replace:: |linkedin|
.. _linkedin_xiaorui: https://www.linkedin.com/in/xiaorui-dong/
.. |linkedin_lucky| replace:: |linkedin|
.. _linkedin_lucky: https://www.linkedin.com/in/lagnajit-pattanaik-94a564108/
.. |linkedin_shihcheng| replace:: |linkedin|
.. _linkedin_shihcheng: https://www.linkedin.com/in/shih-cheng-li-564006207/
.. |linkedin_kevin| replace:: |linkedin|
.. _linkedin_kevin: https://www.linkedin.com/in/kspiekermann/
.. |linkedin_haowei| replace:: |linkedin|
.. _linkedin_haowei: https://www.linkedin.com/in/hao-wei-pang/
.. |linkedin_jwz| replace:: |linkedin|
.. _linkedin_jwz: https://www.linkedin.com/in/mitjonathanzheng/
.. |linkedin_bill| replace:: |linkedin|
.. _linkedin_bill: https://www.linkedin.com/in/william-green-63a9a218/

.. |google_scholar| image:: _static/Google_Scholar_icon.svg
    :class: social-icon
.. |gs_xiaorui| replace:: |google_scholar|
.. _gs_xiaorui: https://scholar.google.com/citations?hl=en&user=r5Wz41EAAAAJ
.. |gs_lucky| replace:: |google_scholar|
.. _gs_lucky: https://scholar.google.com/citations?hl=en&user=bVT6lpwAAAAJ
.. |gs_shihcheng| replace:: |google_scholar|
.. _gs_shihcheng: https://scholar.google.com/citations?hl=en&user=kc_rvjoAAAAJ
.. |gs_kevin| replace:: |google_scholar|
.. _gs_kevin: https://scholar.google.com/citations?hl=en&user=qg2LmbgAAAAJ
.. |gs_haowei| replace:: |google_scholar|
.. _gs_haowei: https://scholar.google.com/citations?hl=en&user=hmkEmtcAAAAJ
.. |gs_jwz| replace:: |google_scholar|
.. _gs_jwz: https://scholar.google.com/citations?user=lVcULZwAAAAJ&hl=en
.. |gs_bill| replace:: |google_scholar|
.. _gs_bill: https://scholar.google.com/citations?hl=en&user=PGQTLWwAAAAJ
