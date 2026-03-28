"""Test that docs/input-parameter.md matches the rendered output of render_input_parameter_doc().

If this test fails, regenerate the docs by running:
    python -m deepks.tools.sync_input_parameter_docs
"""

from pathlib import Path


_DOCS_PATH = Path(__file__).parents[3] / 'docs' / 'input-parameter.md'


def test_input_parameter_docs_not_drifted():
    """Fail if docs/input-parameter.md has drifted from render_input_parameter_doc()."""
    from deepks.io.input import render_input_parameter_doc

    expected = render_input_parameter_doc()

    if not _DOCS_PATH.exists():
        raise AssertionError(
            f"{_DOCS_PATH} does not exist.\n"
            "Run: python -m deepks.tools.sync_input_parameter_docs"
        )

    actual = _DOCS_PATH.read_text(encoding='utf-8')

    assert actual == expected, (
        f"{_DOCS_PATH} has drifted from the rendered output of "
        "render_input_parameter_doc().\n"
        "To sync, run: python -m deepks.tools.sync_input_parameter_docs"
    )
