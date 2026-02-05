import unittest

from easy_as_abc import SolveStatus, _find_best_hint, parse_ascii, solve_puzzle

SAMPLE_UNIQUE_PUZZLE = """\
5 A-C
.B..CC.
A......
......A
.......
......B
B......
....BA.
"""

MULTIPLE_PUZZLE = """\
2 A-B
....
....
....
....
"""

UNSAT_PUZZLE = """\
2 A-B
....
.A..
.A..
....
"""


class EasyAsABCTests(unittest.TestCase):
    def test_solve_sample_unique(self) -> None:
        puzzle = parse_ascii(SAMPLE_UNIQUE_PUZZLE)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.UNIQUE)
        self.assertIsNotNone(result.grid)

    def test_hint_unique_returns_structured_hint(self) -> None:
        puzzle = parse_ascii(SAMPLE_UNIQUE_PUZZLE)
        hint = _find_best_hint(puzzle)
        if hint is None:
            raise AssertionError("Expected a hint for a unique puzzle.")

        self.assertTrue(0 <= hint.row < puzzle.size)
        self.assertTrue(0 <= hint.col < puzzle.size)
        self.assertEqual(puzzle.grid[hint.row + 1][hint.col + 1], ".")
        self.assertIn(hint.value, [*puzzle.letters, "x"])
        self.assertGreater(hint.score, 0)
        self.assertEqual(hint.score, len(hint.core_clues) + len(hint.core_givens))

    def test_hint_multiple_no_hint(self) -> None:
        puzzle = parse_ascii(MULTIPLE_PUZZLE)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.MULTIPLE)
        hint = _find_best_hint(puzzle)
        self.assertIsNone(hint)

    def test_hint_unsat(self) -> None:
        puzzle = parse_ascii(UNSAT_PUZZLE)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.UNSAT)

    def test_parse_rejects_invalid_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "First line must be: N A-C"):
            parse_ascii(
                """\
2A-B
....
....
....
....
"""
            )

    def test_parse_rejects_descending_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "Range start must be <= range end."):
            parse_ascii("2 B-A")

    def test_parse_rejects_range_not_starting_at_a(self) -> None:
        with self.assertRaisesRegex(ValueError, "Range must start at A."):
            parse_ascii(
                """\
3 B-D
.....
.....
.....
.....
.....
"""
            )

    def test_parse_rejects_non_dot_corner(self) -> None:
        with self.assertRaisesRegex(ValueError, "Corner cells must be"):
            parse_ascii(
                """\
2 A-B
A...
....
....
....
"""
            )

    def test_parse_rejects_top_bottom_x_clue(self) -> None:
        with self.assertRaisesRegex(ValueError, "Top/bottom clues must be"):
            parse_ascii(
                """\
2 A-B
.x..
....
....
....
"""
            )

    def test_parse_rejects_left_right_x_clue(self) -> None:
        with self.assertRaisesRegex(ValueError, "Left/right clues must be"):
            parse_ascii(
                """\
2 A-B
....
x...
....
....
"""
            )


if __name__ == "__main__":
    unittest.main()
