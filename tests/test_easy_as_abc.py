import unittest

from easy_as_abc import SolveStatus, _find_best_hint, parse_ascii, solve_puzzle


class EasyAsABCTests(unittest.TestCase):
    def test_solve_sample_unique(self) -> None:
        text = """\
5 A-C
.B..CC.
A......
......A
.......
......B
B......
....BA.
"""
        puzzle = parse_ascii(text)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.UNIQUE)
        self.assertIsNotNone(result.grid)

    def test_hint_unique_returns_hint(self) -> None:
        text = """\
5 A-C
.B..CC.
A......
......A
.......
......B
B......
....BA.
"""
        puzzle = parse_ascii(text)
        hint = _find_best_hint(puzzle)
        self.assertIsNotNone(hint)

    def test_hint_multiple_no_hint(self) -> None:
        text = """\
2 A-B
....
....
....
....
"""
        puzzle = parse_ascii(text)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.MULTIPLE)
        hint = _find_best_hint(puzzle)
        self.assertIsNone(hint)

    def test_hint_unsat(self) -> None:
        text = """\
2 A-B
....
.A..
.A..
....
"""
        puzzle = parse_ascii(text)
        result = solve_puzzle(puzzle, check_unique=True)
        self.assertEqual(result.status, SolveStatus.UNSAT)


if __name__ == "__main__":
    unittest.main()
