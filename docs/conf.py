# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "VideoMeta"
copyright = "2026, UTS iLab P08"
author = "UTS iLab P08"

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    release = _pkg_version("videometa")
except PackageNotFoundError:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_title = "VideoMeta"
html_short_title = "VideoMeta"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

# MyST: let the README's HTML header and anchors render in the docs.
myst_enable_extensions = ["colon_fence", "html_image", "linkify"]
myst_heading_anchors = 3

# Render the example notebook from its saved outputs; do not execute it on build.
nb_execution_mode = "off"
