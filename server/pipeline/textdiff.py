"""How far apart two pieces of text are.

Two questions, and they are not the same one.

:func:`edit_distance` charges both ends: are these two whole lines the same
line with words changed? That is what asking whether a sentence resembles a
known invention needs.

:func:`substring_distance` charges neither: does this text appear somewhere
inside that one? That is what comparing a committed sentence against the
running text needs, because the running text covers part of the sentence and
everything around it must not count as a difference.

Used by the serving pipeline and by the measurement tooling, which is why it
lives here rather than in either.
"""

from __future__ import annotations


def edit_distance(left: str, right: str) -> int:
    """Plain Levenshtein, both ends anchored."""
    if left == right:
        return 0
    if not left or not right:
        return len(left) or len(right)
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, start=1):
        current = [index]
        for position, other in enumerate(right, start=1):
            current.append(min(
                previous[position] + 1,
                current[position - 1] + 1,
                previous[position - 1] + (character != other),
            ))
        previous = current
    return previous[-1]


def substring_distance(needle: str, haystack: str) -> tuple:
    """Edit distance from ``needle`` to its best-matching span of ``haystack``.

    Start and end are free, so text on either side of the match is not
    charged. Returns the distance and the span it matched.
    """
    n, m = len(needle), len(haystack)
    if n == 0:
        return 0, 0, 0
    if m == 0:
        return n, 0, 0

    # Row 0: matching nothing of the needle costs nothing anywhere, which is
    # what makes the start free.
    prev = [0] * (m + 1)
    prev_start = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        cur_start = [0] * (m + 1)
        for j in range(1, m + 1):
            cost = 0 if needle[i - 1] == haystack[j - 1] else 1
            substitute = prev[j - 1] + cost
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            best = min(substitute, delete, insert)
            cur[j] = best
            if best == substitute:
                cur_start[j] = prev_start[j - 1]
            elif best == delete:
                cur_start[j] = prev_start[j]
            else:
                cur_start[j] = cur_start[j - 1]
        prev, prev_start = cur, cur_start

    end = min(range(m + 1), key=lambda j: (prev[j], j))
    return prev[end], prev_start[end], end


def drift(text: str, reference: str) -> float:
    """Share of ``reference`` that ``text`` did not keep, from 0.0 to 1.0.

    0.0 means the reference appears in the text verbatim. 1.0 means nothing
    of it survived. Both sides must already be normalised - this compares
    characters, and it is not the place to decide what counts as one.
    """
    if not reference:
        return 0.0
    distance, _start, _end = substring_distance(reference, text)
    return distance / len(reference)
