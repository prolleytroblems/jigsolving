from solvent import Solvent
from pathlib import Path
from processing_funcs import *

"""solver = Solvent()
puzzle={"./tests/blur-1/puzzle.jpg": "./tests/puzzle.jpg"}
solver.load_puzzles(puzzle)
solver.solve_loaded(out_dir = "./tests/test/", pooling=1, generations=400, threshold =0.95)"""

USE NOT 5x5 SPLIT

folders = []
folders += ["blur-"+str(i) for i in [1,2,3,4]]
folders += ["brightness-"+str(i) for i in [25,50,75,100]]
folders += ["gaussian-"+str(i) for i in [5,10,15,20]]
folders += ["motion-"+str(i) for i in [5,9,13,17]]
basepath=Path("./tests/")
print(folders)

processing_functions = {"Shape_avg_x":lambda x,y: shape_avg(x, 0), "Shape_avg_y":lambda x,y: shape_avg(x, 1),
                        "Shape_stdev_x":lambda x,y: shape_stdev(x, 0), "Shape_stdev_y":lambda x,y: shape_stdev(x, 0),
                        "Area_avg":lambda x,y: area_avg(x), "Area_stdev":lambda x,y: area_stdev(x), "Color_avg_R":lambda x,y: color_avg(y, 2),
                        "Color_avg_G":lambda x,y: color_avg(y, 1), "Color_avg_B":lambda x,y: color_avg(y, 0),
                        "Color_avg_all":lambda x,y: color_avg(y, (0,1,2)), "Color_stdev_R":lambda x,y: color_stdev(y, 2),
                        "Color_stdev_G":lambda x,y: color_stdev(y, 1), "Color_stdev_B":lambda x,y: color_stdev(y, 0),
                        "Color_stdev_all":lambda x,y: color_stdev(y, (0,1,2)), "Piece_count": lambda x,y:len(x)}


solver = Solvent()
solver.config(processing_functions)

solvelist=[("blur-", [4]), ("brightness-", [25,50,75,100]), ("gaussian-", [5,10,15,20]), ("motion-", [5,9,13,17])]

for distortion, devs in solvelist:
    folders=[distortion + str(i) for i in devs]
    for folder in folders:
        solver.load_dir(basepath / folder, basepath)
        solver.solve_loaded(out_dir = basepath / ("out"+folder), pooling=1, generations=300,
                            threshold =[0.3,0.8], max_tries=20, constants={"Distortion": "blur", "D_intensity":folder[-1]})
