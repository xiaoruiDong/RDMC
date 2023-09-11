RDMC Documentation
===================

**RDMC (Reaction Data and Molecular Conformer)** is an open-source lightweight software package specialized in handling Reaction Data and Molecular (including transition states) Conformers.

It contains various modules and classes (e.g., ``RDKitMol``, ``Reaction``, ``view``) helpful for relevant tasks to make conversion, visualization, manipulation, and analysis of molecules easier.
It also provides solutions to pipelining tasks to achieve high-throughput generating and processing of large amount of molecule/reaction data. It is written in Python and has dependencies only
on popular packages (i.e., ``numpy``, ``scipy``, ``matplotlib``, ``rdkit``, ``openbabel``, ``py3dmol``, ``ase``, ``networkx``, ``cclib``), and you can easily incorporate it into your own Python scripts.

The source code of the RDMC software package is hosted on GitHub, and its binary distribution is available on Anaconda Cloud. The easiest way to install RDMC is to use ``conda`` or ``mamba``::

    conda install -c xiaoruidong rdmc

Or

.. code-block:: bash

    mamba install -c xiaoruidong rdmc

``conda`` can be installed by via `Anaconda <https://www.anaconda.com/download/>`_ and, ``mamba`` can be installed via `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_.

You can also install RDMC from the source code:

.. code-block:: bash

    git clone https://github.com/xiaoruidong/rdmc
    cd RDMC
    conda env create -f environment.yml
    conda activate rdmc
    python -m pip install --no-deps -vv ./

To start with, simply try:

.. code-block:: python

    from rdmc import RDKitMol, Reaction
    mol = RDKitMol('CCO')
    rxn = Reaction('CCO>>CC(=O)O')

And see what the ``mol`` and ``rxn`` are capable of! The full lists of APIs of :obj:`RDKitMol <rdmc.mol.RDKitMol>` and :obj:`Reaction <rdmc.reaction.Reaction>` are provided in this documentation.

We also provided a few notebooks (available in ``\ipython`` and `Colab <https://drive.google.com/drive/folders/1bHSChJOocycE5ZfbnGkocX4-OFVDhShW?usp=sharing>`_) to demonstrate the usage of RDMC. Please feel invited to try them out!

RDMC is developed by

- Xiaorui Dong (|github_xiaorui|_ \| |linkedin_xiaorui|_ \| |gs_xiaorui|_),
- Dr. Lagnajit Pattanaik (|github_lucky|_ \| |linkedin_lucky|_ \| |gs_lucky|_),
- Dr. Shih-Cheng Li (|github_shihcheng|_ \| |linkedin_shihcheng|_ \| |gs_shihcheng|_),
- Dr. Kevin Spiekermann (|github_kevin|_ \| |linkedin_kevin|_ \| |gs_kevin|_),
- Hao-Wei Pang (|github_haowei|_ \| |linkedin_haowei|_ \| |gs_haowei|_),
- Prof. William H. Green (|linkedin_bill|_ \| |gs_bill|_)

at `Green Research Group <https://greengroup.mit.edu>`_ at `Massachusetts Institute of Technology (MIT) <https://www.mit.edu>`_.
For any questions while using RDMC, please contact us via the `GitHub issue page <https://github.com/xiaoruiDong/RDMC/issues>`_ or email us at `rdmc_dev@mit.edu <mailto:rdmc@mit.edu>`_.

Contents
========
.. toctree::
   :maxdepth: 2

   reference/rdmc
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
.. |gs_bill| replace:: |google_scholar|
.. _gs_bill: https://scholar.google.com/citations?hl=en&user=PGQTLWwAAAAJ
