"""Sphinx configuration."""

from dataclasses import asdict
from importlib.metadata import version as _get_version

from pydantic.fields import FieldInfo
from sphinx_design.icons import get_octicon
from sphinxawesome_theme import ThemeOptions

# -- Project information -------------------------------------------------------
project = "CaresMCP"
author = "cair-cas.org.hk"
copyright = "2026, cair-cas.org.hk"

# Version comes from the installed package (this is why the build env must
# install the project, which our uv sync does). Single source of truth.
release = _get_version("caresmcp")
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------
extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google / NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",        # [source] links next to API entries
    "sphinx_autodoc_typehints",   # must come after napoleon
    "sphinx_design",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# -- Autodoc / autosummary -----------------------------------------------------
autosummary_generate = True
autodoc_mock_imports = ["transformers"]
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"      # handled by sphinx-autodoc-typehints
typehints_defaults = "comma"           # show defaults in the param description
always_document_param_types = True


# Read undocumented parameter descriptions from pydantic Field metadata.
def _field_description(metadata: object) -> str | None:
    return metadata.description if isinstance(metadata, FieldInfo) else None


typehints_annotated_description = _field_description

# -- Napoleon (docstring style) ------------------------------------------------
napoleon_google_docstring = True       # set False if you use NumPy style only
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False             # let sphinx-autodoc-typehints do return types
napoleon_attr_annotations = True

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "hi-ml": ("https://hi-ml.readthedocs.io/en/latest/", None),
    "huggingface_hub": ("https://huggingface.co/docs/huggingface_hub/main/en/", None),
    "monai": ("https://monai.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "timm": ("https://huggingface.co/docs/timm/main/en/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}
intersphinx_timeout = 10

# -- reStructuredText ----------------------------------------------------------
# Checkbox substitutions for docstring task lists. Autodoc docstrings are
# parsed as RST, where MyST's tasklist extension cannot see them.
rst_prolog = """
.. |x| raw:: html

   <input class="task-list-item-checkbox" type="checkbox" checked disabled>

.. |o| raw:: html

   <input class="task-list-item-checkbox" type="checkbox" disabled>
"""

# -- MyST ----------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",    # ::: directives
    "deflist",
    "fieldlist",
    "attrs_inline",
    "tasklist",
]
myst_heading_anchors = 3   # auto anchors for h1-h3 so [text](#heading) links work

# -- Bibtex --------------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

# -- Warnings policy -----------------------------------------------------------
# Treat unresolved references as warnings. Turn on once the docs are clean;
# with -W it then fails the build on broken cross-references.
nitpicky = False
# nitpick_ignore = [("py:class", "SomeInternalType")]

# -- Syntax highlighting -------------------------------------------------------
# sphinxawesome-theme appends the dark stylesheet behind prefers-color-scheme.
pygments_style = "default"
pygments_style_dark = "one-dark"

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinxawesome_theme"
html_title = f"{project}"
html_favicon = "./_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_permalinks_icon = "<span>#</span>"

# for more icons, see https://primer.style/octicons/
theme_options = ThemeOptions(
    show_scrolltop=True,
    awesome_external_links=True,
    main_nav_links={"Docs": "index"},
    extra_header_link_icons={
        "GitHub": {
            "link": "https://github.com/Moenupa/caresMCP/",
            "icon": get_octicon("mark-github"),
        },
        "About us": {
            "link": "https://cair-cas.org.hk",
            "icon": get_octicon("info"),
        },
    },
    logo_light="./_static/favicon.ico",
    logo_dark="./_static/favicon.ico",
)

html_theme_options = asdict(theme_options)
