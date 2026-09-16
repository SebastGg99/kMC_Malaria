# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MalariaProject - Crystal Growth with kMC method'
copyright = '2026, Sebastian Gaviria Giraldo - Hernan Salinas - German Torres'
author = 'Sebastian Gaviria Giraldo - Hernan Salinas - German Torres'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_rtd_theme

extensions = [
    'myst_parser',
    'sphinx_rtd_theme',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# extensions = []

# templates_path = ['_templates']
# exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = '_static/logo_new.png'
html_favicon = '_static/icono.ico'
html_theme_options = {'logo_only': True}
