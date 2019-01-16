from solvent import Solvent

solver = Solvent()
puzzle={r".\tests\samples\00a0c5ea5b4f3989.jpg": r".\tests\00a0c5ea5b4f3989.jpg"}
solver.load_puzzles(puzzle)
solver.solve_loaded()
