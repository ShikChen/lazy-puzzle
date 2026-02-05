from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import re
import signal
from typing import Iterable, TextIO

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
class SolverContext:
    solver: z3.Solver
    cells: list[list[z3.IntNumRef]]


@dataclass(frozen=True)
class ClueView:
    ref: ClueRef
    symbol: str
    cells: list[z3.IntNumRef]


@dataclass(frozen=True)
class AssumptionSolver:
    context: SolverContext
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
    if letters[0] != "A":
        raise ValueError("Range must start at A.")
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


def _create_cells(size: int) -> list[list[z3.IntNumRef]]:
    return [[z3.Int(f"cell_{r}_{c}") for c in range(size)] for r in range(size)]


def _add_value_domain_constraints(
    solver: z3.Solver, cells: list[list[z3.IntNumRef]], max_value: int
) -> None:
    for row in cells:
        for cell in row:
            solver.add(cell >= 0, cell <= max_value)


def _add_distribution_constraints(
    solver: z3.Solver, cells: list[list[z3.IntNumRef]], letter_count: int
) -> None:
    """Constrain each row/column to contain every letter index exactly once."""
    size = len(cells)
    for row in cells:
        for letter_idx in range(1, letter_count + 1):
            solver.add(z3.Sum([z3.If(cell == letter_idx, 1, 0) for cell in row]) == 1)

    for col in range(size):
        for letter_idx in range(1, letter_count + 1):
            solver.add(
                z3.Sum(
                    [
                        z3.If(cells[row_index][col] == letter_idx, 1, 0)
                        for row_index in range(size)
                    ]
                )
                == 1
            )


def _create_solver_context(puzzle: Puzzle, *, unsat_core: bool) -> SolverContext:
    cells = _create_cells(puzzle.size)
    solver = z3.Solver()
    if unsat_core:
        solver.set(unsat_core=True)
        solver.set("core.minimize", True)
    _add_value_domain_constraints(solver, cells, len(puzzle.letters))
    _add_distribution_constraints(solver, cells, len(puzzle.letters))
    return SolverContext(solver=solver, cells=cells)


@lru_cache(maxsize=None)
def _symbol_to_value(symbol: str) -> int:
    if symbol == "x":
        return 0
    return ord(symbol) - ord("A") + 1


def _add_fixed_cells_constraints(context: SolverContext, puzzle: Puzzle) -> None:
    for row in range(puzzle.size):
        for col in range(puzzle.size):
            symbol = puzzle.grid[row + 1][col + 1]
            if symbol == ".":
                continue
            context.solver.add(context.cells[row][col] == _symbol_to_value(symbol))


def _add_fixed_cells_assumptions(
    context: SolverContext, puzzle: Puzzle
) -> dict[CellRef, z3.BoolRef]:
    given_literals: dict[CellRef, z3.BoolRef] = {}
    for row in range(puzzle.size):
        for col in range(puzzle.size):
            symbol = puzzle.grid[row + 1][col + 1]
            if symbol == ".":
                continue
            literal = z3.Bool(f"given_{row}_{col}")
            given_literals[CellRef(row, col)] = literal
            context.solver.add(
                z3.Implies(
                    literal,
                    context.cells[row][col] == _symbol_to_value(symbol),
                )
            )
    return given_literals


def _collect_clue_views(
    puzzle: Puzzle, cells: list[list[z3.IntNumRef]]
) -> list[ClueView]:
    size = puzzle.size
    top = puzzle.grid[0][1 : size + 1]
    bottom = puzzle.grid[size + 1][1 : size + 1]
    left = [puzzle.grid[row][0] for row in range(1, size + 1)]
    right = [puzzle.grid[row][size + 1] for row in range(1, size + 1)]
    clues_by_side: dict[ClueSide, list[str]] = {
        ClueSide.TOP: top,
        ClueSide.BOTTOM: bottom,
        ClueSide.LEFT: left,
        ClueSide.RIGHT: right,
    }

    def clue_cells(side: ClueSide, index: int) -> list[z3.IntNumRef]:
        match side:
            case ClueSide.TOP:
                return [cells[row][index] for row in range(size)]
            case ClueSide.BOTTOM:
                return [cells[-1 - row][index] for row in range(size)]
            case ClueSide.LEFT:
                return cells[index]
            case ClueSide.RIGHT:
                return list(reversed(cells[index]))

    clue_views: list[ClueView] = []
    for side, symbols in clues_by_side.items():
        for index, symbol in enumerate(symbols):
            if symbol == ".":
                continue
            clue_views.append(
                ClueView(
                    ref=ClueRef(side, index),
                    symbol=symbol,
                    cells=clue_cells(side, index),
                )
            )

    return clue_views


def _add_clue_constraints(context: SolverContext, puzzle: Puzzle) -> None:
    for clue in _collect_clue_views(puzzle, context.cells):
        context.solver.add(
            _first_visible_constraint(
                clue.cells,
                _symbol_to_value(clue.symbol),
            )
        )


def _clue_literal_name(clue_ref: ClueRef) -> str:
    return f"clue_{clue_ref.side.value}_{clue_ref.index}"


def _add_clue_assumptions(
    context: SolverContext, puzzle: Puzzle
) -> dict[ClueRef, z3.BoolRef]:
    clue_literals: dict[ClueRef, z3.BoolRef] = {}
    for clue in _collect_clue_views(puzzle, context.cells):
        literal = z3.Bool(_clue_literal_name(clue.ref))
        clue_literals[clue.ref] = literal
        context.solver.add(
            z3.Implies(
                literal,
                _first_visible_constraint(
                    clue.cells,
                    _symbol_to_value(clue.symbol),
                ),
            )
        )
    return clue_literals


def _build_solver(puzzle: Puzzle) -> SolverContext:
    context = _create_solver_context(puzzle, unsat_core=False)
    _add_fixed_cells_constraints(context, puzzle)
    _add_clue_constraints(context, puzzle)
    return context


def _build_solver_with_assumptions(puzzle: Puzzle) -> AssumptionSolver:
    context = _create_solver_context(puzzle, unsat_core=True)
    given_literals = _add_fixed_cells_assumptions(context, puzzle)
    clue_literals = _add_clue_assumptions(context, puzzle)
    return AssumptionSolver(
        context=context,
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
    context = _build_solver(puzzle)
    if context.solver.check() != z3.sat:
        return SolveResult(status=SolveStatus.UNSAT, grid=None)

    model = context.solver.model()
    solution = _solution_grid(puzzle, context.cells, model)
    status = SolveStatus.SAT

    if check_unique:
        differences: list[z3.BoolRef] = []
        for row in context.cells:
            for cell in row:
                differences.append(cell != model.evaluate(cell, model_completion=True))
        context.solver.push()
        context.solver.add(z3.Or(differences))
        status = (
            SolveStatus.MULTIPLE
            if context.solver.check() == z3.sat
            else SolveStatus.UNIQUE
        )
        context.solver.pop()

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


def _read_puzzle_text(file_obj: TextIO) -> str:
    if file_obj.isatty():
        return SAMPLE_PUZZLE
    text = file_obj.read()
    if not text.strip():
        text = SAMPLE_PUZZLE

    return text


def _parse_or_click_error(text: str) -> Puzzle:
    try:
        return parse_ascii(text)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _is_better_hint(candidate: HintResult, current: HintResult | None) -> bool:
    if current is None:
        return True
    return (candidate.score, candidate.row, candidate.col) < (
        current.score,
        current.row,
        current.col,
    )


def _find_best_hint(
    puzzle: Puzzle, solve_result: SolveResult | None = None
) -> HintResult | None:
    if solve_result is None:
        solve_result = solve_puzzle(puzzle, check_unique=True)
    if solve_result.status != SolveStatus.UNIQUE or solve_result.grid is None:
        return None

    model_grid = solve_result.grid
    bundle = _build_solver_with_assumptions(puzzle)
    assumptions = [*bundle.clue_literals.values(), *bundle.given_literals.values()]
    solver = bundle.context.solver
    cells = bundle.context.cells
    best: HintResult | None = None

    for row in range(puzzle.size):
        for col in range(puzzle.size):
            if puzzle.grid[row + 1][col + 1] != ".":
                continue
            forced_symbol = model_grid[row + 1][col + 1]
            forced_value = _symbol_to_value(forced_symbol)

            solver.push()
            solver.add(cells[row][col] != forced_value)
            is_unsat = solver.check(*assumptions) == z3.unsat
            core_set = set(solver.unsat_core()) if is_unsat else set()
            solver.pop()
            if not is_unsat:
                continue

            core_clues = {
                key for key, lit in bundle.clue_literals.items() if lit in core_set
            }
            core_givens = {
                key for key, lit in bundle.given_literals.items() if lit in core_set
            }
            score = len(core_set)
            candidate = HintResult(
                row=row,
                col=col,
                value=forced_symbol,
                score=score,
                core_clues=core_clues,
                core_givens=core_givens,
            )
            if _is_better_hint(candidate, best):
                best = candidate

    return best


@click.group()
@click.option(
    "--file",
    "file_obj",
    type=click.File("r"),
    default="-",
    show_default="stdin",
    help="Read puzzle from file path (defaults to stdin).",
)
@click.pass_context
def cli(ctx: click.Context, file_obj: TextIO) -> None:
    ctx.ensure_object(dict)
    ctx.obj["text"] = _read_puzzle_text(file_obj)


@cli.command()
@click.pass_context
def solve(ctx: click.Context) -> None:
    text = ctx.obj["text"]
    puzzle = _parse_or_click_error(text)

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
    puzzle = _parse_or_click_error(text)

    solve_result = solve_puzzle(puzzle, check_unique=True)
    if solve_result.status != SolveStatus.UNIQUE or solve_result.grid is None:
        click.echo(solve_result.status.value)
        if solve_result.status == SolveStatus.MULTIPLE:
            click.echo("NO HINT")
        return

    hint_result = _find_best_hint(puzzle, solve_result=solve_result)
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
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    cli()
