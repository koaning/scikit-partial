import pytest
from mktestdocs import check_docstring, check_md_file

from skpartial.pipeline import (
    PartialPipeline,
    PartialFeatureUnion,
    make_partial_pipeline,
    make_partial_union,
)


components = [
    PartialPipeline,
    PartialFeatureUnion,
    make_partial_pipeline,
    make_partial_union,
]


@pytest.mark.parametrize("obj", components, ids=lambda d: d.__qualname__)
def test_member(obj):
    """The example snippets must run."""
    check_docstring(obj)


def test_readme_works():
    """The code-blocks must run."""
    check_md_file("README.md", memory=True)
