from solvent import Solvent

solver = Solvent()
puzzle={"./images/samples/sunset.jpg": "./images/sunset.jpg"}
solver.load_puzzles(puzzle)
solver.solve_loaded(out_dir = "./images/out/")
