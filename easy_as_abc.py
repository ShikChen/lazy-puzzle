from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
import sys
from typing import Iterable

import click
import z3

# Easy as ABC (Classic) rules:
# - Fill an NxN grid with letters from the given range (e.g., A-C).
# - Each row and each column contains each letter exactly once.
# - All other cells are empty (output as `x`).
# - Edge clues indicate the first visible letter from that side.
# - Given letters and `x` cells are fixed.

SAMPLE_PUZZLE = """\
5 A-C
.B..CC.
A......
......A
.......
......B
B......
....BA.
"""


@dataclass(frozen=True)
class Puzzle:
    size: int
    letters: list[str]
    grid: list[list[str]]


class SolveStatus(StrEnum):
    SAT = "SAT"
    UNIQUE = "UNIQUE"
    MULTIPLE = "MULTIPLE"
    UNSAT = "UNSAT"


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    grid: list[list[str]] | None


def _parse_range(range_text: str) -> list[str]:
    match = re.fullmatch(r"([A-Za-z])-([A-Za-z])", range_text.strip())
    if not match:
        raise ValueError("Range must look like A-C.")
    start, end = match.group(1).upper(), match.group(2).upper()
    if ord(start) > ord(end):
        raise ValueError("Range start must be <= range end.")
    return [chr(code) for code in range(ord(start), ord(end) + 1)]


def parse_ascii(text: str) -> Puzzle:
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() != ""]
    if not lines:
        raise ValueError("Input is empty.")
    header = lines[0].strip()
    parts = header.split()
    if len(parts) != 2:
        raise ValueError("First line must be: N A-C")
    try:
        size = int(parts[0])
    except ValueError as exc:
        raise ValueError("Grid size must be an integer.") from exc
    if size <= 0:
        raise ValueError("Grid size must be positive.")

    letters = _parse_range(parts[1])
    if len(letters) > size:
        raise ValueError("Range length cannot exceed grid size.")
    expected_lines = size + 2
    grid_lines = lines[1:]
    if len(grid_lines) != expected_lines:
        raise ValueError(
            f"Expected {expected_lines} grid lines, got {len(grid_lines)}."
        )

    grid: list[list[str]] = []
    for row_index, raw in enumerate(grid_lines):
        line = raw.strip()
        if len(line) != expected_lines:
            raise ValueError(
                f"Grid line {row_index + 1} must be length {expected_lines}."
            )
        row_cells: list[str] = []
        for ch in line:
            if ch == ".":
                row_cells.append(ch)
            elif ch in ("x", "X"):
                row_cells.append("x")
            elif ch.upper() in letters:
                row_cells.append(ch.upper())
            else:
                raise ValueError(f"Invalid character '{ch}' in grid.")
        grid.append(row_cells)

    corners = [
        grid[0][0],
        grid[0][size + 1],
        grid[size + 1][0],
        grid[size + 1][size + 1],
    ]
    if any(ch != "." for ch in corners):
        raise ValueError("Corner cells must be '.'.")

    for col in range(1, size + 1):
        if grid[0][col] == "x" or grid[size + 1][col] == "x":
            raise ValueError("Top/bottom clues must be letters or '.'.")
    for row_idx in range(1, size + 1):
        if grid[row_idx][0] == "x" or grid[row_idx][size + 1] == "x":
            raise ValueError("Left/right clues must be letters or '.'.")

    return Puzzle(size=size, letters=letters, grid=grid)


def _first_visible_constraint(cells: Iterable[z3.IntNumRef], value: int) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    prefix_clear = z3.BoolVal(True)
    for cell in cells:
        clauses.append(z3.And(prefix_clear, cell == value))
        prefix_clear = z3.And(prefix_clear, cell == 0)
    return z3.Or(clauses)


def _build_solver(puzzle: Puzzle) -> tuple[z3.Solver, list[list[z3.IntNumRef]]]:
    size = puzzle.size
    letters = puzzle.letters
    letter_count = len(letters)
    letter_to_idx = {letter: idx + 1 for idx, letter in enumerate(letters)}

    cells: list[list[z3.IntNumRef]] = [
        [z3.Int(f"cell_{r}_{c}") for c in range(size)] for r in range(size)
    ]
    solver = z3.Solver()

    for r in range(size):
        for c in range(size):
            solver.add(cells[r][c] >= 0, cells[r][c] <= letter_count)

    for r in range(size):
        for letter_idx in range(1, letter_count + 1):
            solver.add(
                z3.Sum([z3.If(cells[r][c] == letter_idx, 1, 0) for c in range(size)])
                == 1
            )

    for c in range(size):
        for letter_idx in range(1, letter_count + 1):
            solver.add(
                z3.Sum([z3.If(cells[r][c] == letter_idx, 1, 0) for r in range(size)])
                == 1
            )

    for r in range(size):
        for c in range(size):
            token = puzzle.grid[r + 1][c + 1]
            if token == "x":
                solver.add(cells[r][c] == 0)
            elif token != ".":
                solver.add(cells[r][c] == letter_to_idx[token])

    top = puzzle.grid[0][1 : size + 1]
    bottom = puzzle.grid[size + 1][1 : size + 1]
    left = [puzzle.grid[r][0] for r in range(1, size + 1)]
    right = [puzzle.grid[r][size + 1] for r in range(1, size + 1)]

    for c, clue in enumerate(top):
        if clue != ".":
            solver.add(
                _first_visible_constraint(
                    [cells[r][c] for r in range(size)], letter_to_idx[clue]
                )
            )
    for c, clue in enumerate(bottom):
        if clue != ".":
            solver.add(
                _first_visible_constraint(
                    [cells[r][c] for r in reversed(range(size))],
                    letter_to_idx[clue],
                )
            )
    for r, clue in enumerate(left):
        if clue != ".":
            solver.add(_first_visible_constraint(cells[r], letter_to_idx[clue]))
    for r, clue in enumerate(right):
        if clue != ".":
            solver.add(
                _first_visible_constraint(list(reversed(cells[r])), letter_to_idx[clue])
            )

    return solver, cells


def _solution_grid(
    puzzle: Puzzle, cells: list[list[z3.IntNumRef]], model: z3.ModelRef
) -> list[list[str]]:
    size = puzzle.size
    letters = puzzle.letters
    output = [row[:] for row in puzzle.grid]
    for r in range(size):
        for c in range(size):
            value = model.evaluate(cells[r][c], model_completion=True).as_long()
            if value == 0:
                output[r + 1][c + 1] = "x"
            else:
                output[r + 1][c + 1] = letters[value - 1]
    return output


def solve_puzzle(puzzle: Puzzle, check_unique: bool = True) -> SolveResult:
    solver, cells = _build_solver(puzzle)
    if solver.check() != z3.sat:
        return SolveResult(status=SolveStatus.UNSAT, grid=None)

    model = solver.model()
    solution = _solution_grid(puzzle, cells, model)
    status = SolveStatus.SAT

    if check_unique:
        differences: list[z3.BoolRef] = []
        for row in cells:
            for cell in row:
                differences.append(cell != model.evaluate(cell, model_completion=True))
        solver.push()
        solver.add(z3.Or(differences))
        status = (
            SolveStatus.MULTIPLE if solver.check() == z3.sat else SolveStatus.UNIQUE
        )
        solver.pop()

    return SolveResult(status=status, grid=solution)


def solve_text(text: str, check_unique: bool = True) -> SolveResult:
    puzzle = parse_ascii(text)
    return solve_puzzle(puzzle, check_unique=check_unique)


def _format_clue_line(clues: list[str], size: int) -> str:
    chars = [" "] * (size * 4 + 1)
    for idx, clue in enumerate(clues):
        if clue != ".":
            chars[2 + idx * 4] = clue
    return (" " * 4 + "".join(chars)).rstrip()


def format_solution(puzzle: Puzzle, grid: list[list[str]]) -> str:
    size = puzzle.size
    top = puzzle.grid[0][1 : size + 1]
    bottom = puzzle.grid[size + 1][1 : size + 1]
    left = [puzzle.grid[r][0] for r in range(1, size + 1)]
    right = [puzzle.grid[r][size + 1] for r in range(1, size + 1)]

    border = "+" + "+".join(["---"] * size) + "+"
    lines: list[str] = []
    lines.append(_format_clue_line(top, size))
    lines.append(f"{'':>3} {border}")
    for r in range(size):
        left_clue = left[r] if left[r] != "." else ""
        right_clue = right[r] if right[r] != "." else ""
        row_cells = (
            "|" + "|".join(f" {cell} " for cell in grid[r + 1][1 : size + 1]) + "|"
        )
        line = f"{left_clue:>3} {row_cells}"
        if right_clue:
            line += f" {right_clue}"
        lines.append(line.rstrip())
        lines.append(f"{'':>3} {border}")
    lines.append(_format_clue_line(bottom, size))
    return "\n".join(lines).rstrip()


@click.command()
@click.option(
    "--file", "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--stdin", "force_stdin", is_flag=True, help="Read puzzle from stdin.")
def cli(file_path: Path | None, force_stdin: bool) -> None:
    if file_path and force_stdin:
        raise click.ClickException("Use either --file or --stdin, not both.")

    if file_path:
        text = file_path.read_text()
    else:
        if force_stdin or not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            text = ""

    if not text.strip():
        text = SAMPLE_PUZZLE

    try:
        puzzle = parse_ascii(text)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    result = solve_puzzle(puzzle, check_unique=True)
    if result.status == SolveStatus.UNSAT or result.grid is None:
        click.echo(SolveStatus.UNSAT.value)
        return

    click.echo(result.status.value)
    click.echo(format_solution(puzzle, result.grid))


if __name__ == "__main__":
    cli()
