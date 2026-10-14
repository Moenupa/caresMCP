import re

import pytest

from caresmcp.prompts.permodality import GUIDELINES, INSTRUCTIONS

BULLET_POINT_STRUCTURE = re.compile(r"^-?[^\n]*(?:\n- [^\n]*)*$")


# instructions should not contain more than one consecutive \s
@pytest.mark.parametrize("instruction", list(INSTRUCTIONS), ids=lambda item: item.name)
def test_instruction_format(instruction: INSTRUCTIONS):
    assert not re.search(r"\s{2,}", instruction)
    assert not re.search(r"\n", instruction)


@pytest.mark.parametrize("guideline", list(GUIDELINES), ids=lambda item: item.name)
def test_guideline_format(guideline: GUIDELINES) -> None:
    """Every GUIDELINES enum value is a markdown bullet containing a link and label."""
    assert re.fullmatch(
        BULLET_POINT_STRUCTURE,
        guideline.value,
    ), f"{guideline.name} does not match the expected guideline format"
