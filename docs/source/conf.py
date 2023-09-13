# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'RDMC'
copyright = f'2020-{datetime.datetime.now().year}, Xiaorui Dong'
author = 'Xiaorui Dong, Lagnajit Pattanaik, Shih-Cheng Li, ' \
         'Kevin Spiekermann, Hao-Wei Pang, and William H. Green'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # automatically add docstring
    'sphinx.ext.napoleon',  # support for NumPy and Google style docstrings
    'sphinx.ext.todo',  # include to do
    'sphinx.ext.coverage',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates',]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xiaoruidong/rdmc",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ]
}

# Make the `Returns` in docstring behave like `Args`
napoleon_custom_sections = [('Returns', 'params_style')]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/RDMC_icon.svg'


# -- Packages that are not importable from the default environment ---------

# Most of the imports are located in the external module
autodoc_mock_imports = ['rmgpy',]
