# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())
from ops.cyclotron.analysis import __version__

# -- Project information -----------------------------------------------------

project = 'LBNL 88-Inch Cyclotron Analysis Codes'
copyright = '2025, Jessica Rehak'
author = 'Jessica Rehak'
release = __version__

# -- General configuration ---------------------------------------------------

default_role = 'py:obj'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []
modindex_common_prefix = ['ops.cyclotron.analysis.']

# EXTENSION OPTIONS
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

autodoc_default_options = {
    'member_order': 'bysource',
    'special_members': '__init__',
    'members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'both' #'description'
autodoc_class_signature = 'separated'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    github_base = 'https://github.com/lbnl-88ops/'
    repo = 'cyclotron.analysis'
    branch = 'main'
    return '{base}{repo}/tree/{branch}/{filename}.py'.format(
        base=github_base, 
        repo=repo, 
        branch=branch, 
        filename=filename)

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'logo_only': True
}
