# Configuration file for the Sphinx documentation builder.

autodoc_mock_imports = ['rdkit', 'umap', 'sklearn','selfies','scipy']

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# This avoids the methods to be presented alphabetically, which is the default
autodoc_member_order = 'bysource'

# -- Path setup

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../macaw/"))

# -- Project information

exec(open('../../macaw/version.py').read())
project = 'MACAW'
copyright = '2021, LBNL'
author = 'Vincent'

version = __version__

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
