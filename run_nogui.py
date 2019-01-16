from solvent import Solvent

solver = Solvent()
puzzle={"./images/samples/sunset.jpg": "./images/sunset.jpg"}
solver.load_dir("./images/samples/", "./images/")
solver.solve_loaded(out_dir = "./images/out/")
