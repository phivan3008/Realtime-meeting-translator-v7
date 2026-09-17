"""Compare client debug logs of the same meeting, side by side.

    python -m server.analysis.compare_logs recordings/meeting-A.debug.txt \
        recordings/meeting-B.debug.txt [...]

The first log is the reference. Every other one is checked to be the same
meeting before any number is printed next to it, because comparing two
different recordings line by line turns noise into a finding - this project
has made that mistake once already.

Pure text: runs anywhere, no audio and no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.analysis.debuglog import measure, parse, same_meeting  # noqa: E402

#: The rows printed, in order, with how each is read.
ROWS = (
    ("sentences", "{:d}"),
    ("characters", "{:d}"),
    ("japanese_sentences", "{:d}"),
    ("japanese_spaced", "{:d}"),
    ("with_repeats", "{:d}"),
    ("repeat_total", "{:d}"),
    ("empty_share", "{:.1%}"),
    ("refused_translations", "{:d}"),
    ("near_block_list", "{:d}"),
    ("mean_rewrite", "{:.3f}"),
    ("far_from_partial", "{:d}"),
    ("language_disagrees", "{:d}"),
    ("mixed_language", "{:d}"),
    ("lost_turns", "{:d}"),
    ("update_survival", "{:.1%}"),
    ("updates_wiping_half", "{:.1%}"),
)


def table(paths: list) -> str:
    runs = [parse(Path(path)) for path in paths]
    numbers = [measure(run) for run in runs]
    for run, found in zip(runs, numbers):
        found["empty_share"] = (
            sum(1 for final in run.finals if not final.text.strip())
            / max(len(run.finals), 1))

    width = max(len(name) for name, _ in ROWS) + 2
    column = 22
    lines = ["".ljust(width) + "".join(run.name[-column + 1:].rjust(column)
                                        for run in runs)]
    for name, shape in ROWS:
        lines.append(name.ljust(width) + "".join(
            shape.format(found[name]).rjust(column) for found in numbers))

    lines.append("")
    reference = runs[0]
    for run in runs[1:]:
        verdict = same_meeting(reference, run)
        lines.append(
            f"{run.name}: {'same meeting' if verdict['same'] else 'NOT the same meeting'}"
            f" as {reference.name} (offset {verdict['offset']:+.1f} s, "
            f"{verdict['votes']}/{verdict['shared']} shared words agree)")
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("logs", nargs="+", help="client *.debug.txt files")
    args = parser.parse_args(argv)
    print(table(args.logs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
