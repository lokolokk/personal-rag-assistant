"""Conservative textual evidence for an introduced, multi-item list."""

import re
from collections.abc import Sequence

_NUMBERED_ITEM = re.compile(r"^([0-9]{1,6})[.)]\s+\S")
_BULLET_ITEM = re.compile(r"^[-*•]\s+\S")


def introduced_list_items(lines: Sequence[str], preceding_text: str = "") -> set[int]:
    """Recognize sequential numbered items after a colon, including wrapped items.

    A numbered line alone is ambiguous: genuine academic headings must survive.
    Require an introduction and at least two consecutive item numbers.
    """
    numbered = [
        (i, int(m[1])) for i, line in enumerate(lines) if (m := _NUMBERED_ITEM.match(line.strip()))
    ]
    result: set[int] = set()
    for position, (index, number) in enumerate(numbered):
        before = lines[index - 1] if index else preceding_text
        if not before.rstrip().endswith(":"):
            continue
        group = [index]
        for next_index, next_number in numbered[position + 1 :]:
            if next_number != number + len(group):
                break
            group.append(next_index)
        if len(group) >= 2:
            result.update(group)
    return result


def continues_introduced_list(introduction: str, text: str) -> bool:
    """Require a direct numbered or bulleted list following an introduction."""
    if not introduction.rstrip().endswith(":"):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if 0 in introduced_list_items(lines, introduction):
        return True
    return (
        bool(_BULLET_ITEM.match(lines[0]))
        and sum(bool(_BULLET_ITEM.match(line)) for line in lines) >= 2
    )
