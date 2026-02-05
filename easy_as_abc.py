import z3


def main():
    x = z3.Int("x")
    y = z3.Int("y")
    solver = z3.Solver()
    solver.add(x + y == 10)
    solver.add(x - y == 4)
    if solver.check() == z3.sat:
        model = solver.model()
        print(f"x = {model[x]}, y = {model[y]}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
