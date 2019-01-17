from solvent import Solvent
from pathlib import Path

"""solver = Solvent()
puzzle={"./tests/blur-1/puzzle.jpg": "./tests/puzzle.jpg"}
solver.load_puzzles(puzzle)
solver.solve_loaded(out_dir = "./tests/test/", pooling=1, generations=400, threshold =0.95)"""



folders = []
folders += ["blur-"+str(i) for i in [1,2,3,4]]
"""folders += ["brightness-"+str(i) for i in [25,50,75,100]]
folders += ["gaussian-"+str(i) for i in [5,10,15,20]]
folders += ["motion-"+str(i) for i in [5,9,13,17]]"""
basepath=Path("./tests/")
print(folders)
for folder in folders:
    solver = Solvent()
    solver.load_dir(basepath / folder, basepath)
    solver.solve_loaded(out_dir = basepath / ("out"+folder), pooling=1, generations=200, threshold =0.95)
