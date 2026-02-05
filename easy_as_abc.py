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


class ClueSide(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class CellRef:
    row: int
    col: int


@dataclass(frozen=True)
class ClueRef:
    side: ClueSide
    index: int


@dataclass(frozen=True)
class AssumptionSolver:
    solver: z3.Solver
    cells: list[list[z3.IntNumRef]]
    clue_literals: dict[ClueRef, z3.BoolRef]
    given_literals: dict[CellRef, z3.BoolRef]


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    grid: list[list[str]] | None


@dataclass(frozen=True)
class HintResult:
    row: int
    col: int
    value: str
    score: int
    core_clues: set[ClueRef]
    core_givens: set[CellRef]


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


def _build_solver_with_assumptions(
    puzzle: Puzzle,
) -> AssumptionSolver:
    size = puzzle.size
    letters = puzzle.letters
    letter_count = len(letters)
    letter_to_idx = {letter: idx + 1 for idx, letter in enumerate(letters)}

    cells: list[list[z3.IntNumRef]] = [
        [z3.Int(f"cell_{r}_{c}") for c in range(size)] for r in range(size)
    ]
    solver = z3.Solver()
    solver.set(unsat_core=True)

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

    given_literals: dict[CellRef, z3.BoolRef] = {}
    for r in range(size):
        for c in range(size):
            token = puzzle.grid[r + 1][c + 1]
            if token == ".":
                continue
            lit = z3.Bool(f"given_{r}_{c}")
            given_literals[CellRef(r, c)] = lit
            if token == "x":
                solver.add(z3.Implies(lit, cells[r][c] == 0))
            else:
                solver.add(z3.Implies(lit, cells[r][c] == letter_to_idx[token]))

    clue_literals: dict[ClueRef, z3.BoolRef] = {}
    top = puzzle.grid[0][1 : size + 1]
    bottom = puzzle.grid[size + 1][1 : size + 1]
    left = [puzzle.grid[r][0] for r in range(1, size + 1)]
    right = [puzzle.grid[r][size + 1] for r in range(1, size + 1)]

    for c, clue in enumerate(top):
        if clue != ".":
            lit = z3.Bool(f"clue_top_{c}")
            clue_literals[ClueRef(ClueSide.TOP, c)] = lit
            solver.add(
                z3.Implies(
                    lit,
                    _first_visible_constraint(
                        [cells[r][c] for r in range(size)], letter_to_idx[clue]
                    ),
                )
            )
    for c, clue in enumerate(bottom):
        if clue != ".":
            lit = z3.Bool(f"clue_bottom_{c}")
            clue_literals[ClueRef(ClueSide.BOTTOM, c)] = lit
            solver.add(
                z3.Implies(
                    lit,
                    _first_visible_constraint(
                        [cells[r][c] for r in reversed(range(size))],
                        letter_to_idx[clue],
                    ),
                )
            )
    for r, clue in enumerate(left):
        if clue != ".":
            lit = z3.Bool(f"clue_left_{r}")
            clue_literals[ClueRef(ClueSide.LEFT, r)] = lit
            solver.add(
                z3.Implies(
                    lit, _first_visible_constraint(cells[r], letter_to_idx[clue])
                )
            )
    for r, clue in enumerate(right):
        if clue != ".":
            lit = z3.Bool(f"clue_right_{r}")
            clue_literals[ClueRef(ClueSide.RIGHT, r)] = lit
            solver.add(
                z3.Implies(
                    lit,
                    _first_visible_constraint(
                        list(reversed(cells[r])), letter_to_idx[clue]
                    ),
                )
            )

    return AssumptionSolver(
        solver=solver,
        cells=cells,
        clue_literals=clue_literals,
        given_literals=given_literals,
    )


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


def _format_clue_line(clues: list[str], size: int, highlights: set[int] | None) -> str:
    chars = [" "] * (size * 4 + 1)
    for idx, clue in enumerate(clues):
        if clue == ".":
            continue
        value = clue
        if highlights and idx in highlights:
            value = click.style(value, fg="bright_cyan")
        chars[2 + idx * 4] = value
    return (" " * 4 + "".join(chars)).rstrip()


def format_solution(
    puzzle: Puzzle,
    grid: list[list[str]],
    highlight_cells: set[CellRef] | None = None,
    highlight_clues: set[ClueRef] | None = None,
    hint_cell: CellRef | None = None,
) -> str:
    size = puzzle.size
    top = puzzle.grid[0][1 : size + 1]
    bottom = puzzle.grid[size + 1][1 : size + 1]
    left = [puzzle.grid[r][0] for r in range(1, size + 1)]
    right = [puzzle.grid[r][size + 1] for r in range(1, size + 1)]

    clue_highlights: dict[ClueSide, set[int]] = {}
    if highlight_clues:
        for clue in highlight_clues:
            clue_highlights.setdefault(clue.side, set()).add(clue.index)

    border = "+" + "+".join(["---"] * size) + "+"
    lines: list[str] = []
    lines.append(_format_clue_line(top, size, clue_highlights.get(ClueSide.TOP)))
    lines.append(f"{'':>3} {border}")
    for r in range(size):
        left_clue = left[r] if left[r] != "." else ""
        right_clue = right[r] if right[r] != "." else ""
        left_highlight = (
            highlight_clues is not None and ClueRef(ClueSide.LEFT, r) in highlight_clues
        )
        right_highlight = (
            highlight_clues is not None
            and ClueRef(ClueSide.RIGHT, r) in highlight_clues
        )
        row_tokens: list[str] = []
        for c, cell in enumerate(grid[r + 1][1 : size + 1]):
            coords = CellRef(r, c)
            display = cell
            if cell == "." and not (hint_cell and coords == hint_cell):
                display = " "
            styled = display
            if hint_cell and coords == hint_cell:
                styled = click.style(display, fg="bright_green")
            elif highlight_cells and coords in highlight_cells:
                styled = click.style(display, fg="yellow")
            row_tokens.append(f" {styled} ")
        row_cells = "|" + "|".join(row_tokens) + "|"
        if left_clue:
            left_display = (
                click.style(left_clue, fg="bright_cyan")
                if left_highlight
                else left_clue
            )
            left_field = f"{'':>2}{left_display}"
        else:
            left_field = f"{'':>3}"
        line = f"{left_field} {row_cells}"
        if right_clue:
            right_display = (
                click.style(right_clue, fg="bright_cyan")
                if right_highlight
                else right_clue
            )
            line += f" {right_display}"
        lines.append(line.rstrip())
        lines.append(f"{'':>3} {border}")
    lines.append(_format_clue_line(bottom, size, clue_highlights.get(ClueSide.BOTTOM)))
    return "\n".join(lines).rstrip()


def _read_puzzle_text(file_path: Path | None, force_stdin: bool) -> str:
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

    return text


def _hint_value_symbol(puzzle: Puzzle, value: int) -> str:
    if value == 0:
        return "x"
    return puzzle.letters[value - 1]


def _minimal_core(solver: z3.Solver) -> list[z3.BoolRef]:
    core = list(solver.unsat_core())
    core_set = set(core)
    changed = True
    while changed:
        changed = False
        for lit in list(core_set):
            trial = [a for a in core_set if a is not lit]
            if not trial:
                continue
            if solver.check(*trial) == z3.unsat:
                core_set.remove(lit)
                changed = True
    return [lit for lit in core if lit in core_set]


def _find_best_hint(puzzle: Puzzle) -> HintResult | None:
    solve_result = solve_puzzle(puzzle, check_unique=True)
    if solve_result.status != SolveStatus.UNIQUE or solve_result.grid is None:
        return None

    model_grid = solve_result.grid
    size = puzzle.size
    best: HintResult | None = None

    for r in range(size):
        for c in range(size):
            if puzzle.grid[r + 1][c + 1] != ".":
                continue
            bundle = _build_solver_with_assumptions(puzzle)
            forced_value = model_grid[r + 1][c + 1]
            forced_symbol = forced_value
            forced_int = (
                0 if forced_value == "x" else puzzle.letters.index(forced_value) + 1
            )
            bundle.solver.add(bundle.cells[r][c] != forced_int)
            assumptions = list(bundle.clue_literals.values()) + list(
                bundle.given_literals.values()
            )
            if bundle.solver.check(*assumptions) != z3.unsat:
                continue
            core = _minimal_core(bundle.solver)
            core_set = set(core)
            core_clues = {
                key for key, lit in bundle.clue_literals.items() if lit in core_set
            }
            core_givens = {
                key for key, lit in bundle.given_literals.items() if lit in core_set
            }
            score = len(core)
            candidate = HintResult(
                row=r,
                col=c,
                value=forced_symbol,
                score=score,
                core_clues=core_clues,
                core_givens=core_givens,
            )
            if (
                best is None
                or score < best.score
                or (score == best.score and (r, c) < (best.row, best.col))
            ):
                best = candidate

    return best


@click.group()
@click.option(
    "--file", "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--stdin", "force_stdin", is_flag=True, help="Read puzzle from stdin.")
@click.pass_context
def cli(ctx: click.Context, file_path: Path | None, force_stdin: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["text"] = _read_puzzle_text(file_path, force_stdin)


@cli.command()
@click.pass_context
def solve(ctx: click.Context) -> None:
    text = ctx.obj["text"]
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


@cli.command()
@click.pass_context
def hint(ctx: click.Context) -> None:
    text = ctx.obj["text"]
    try:
        puzzle = parse_ascii(text)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    solve_result = solve_puzzle(puzzle, check_unique=True)
    if solve_result.status != SolveStatus.UNIQUE or solve_result.grid is None:
        click.echo(solve_result.status.value)
        if solve_result.status == SolveStatus.MULTIPLE:
            click.echo("NO HINT")
        return

    hint_result = _find_best_hint(puzzle)
    if hint_result is None:
        click.echo("NO HINT")
        return

    click.echo(SolveStatus.UNIQUE.value)
    click.echo(
        f"HINT {hint_result.row + 1} {hint_result.col + 1} = {hint_result.value} "
        f"(score: {hint_result.score})"
    )
    hint_cell = CellRef(hint_result.row, hint_result.col)
    click.echo(
        format_solution(
            puzzle,
            puzzle.grid,
            highlight_cells=hint_result.core_givens,
            highlight_clues=hint_result.core_clues,
            hint_cell=hint_cell,
        )
    )


if __name__ == "__main__":
    cli()
