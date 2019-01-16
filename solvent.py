from utils import *
from img_recog_numba import full_solve
from piecefinder import PieceFinder
import json
import cv2
import os
from pathlib import Path
from datetime import datetime
from image_obj import PieceCollection, Solution

DEFAULTS = {"debug_mode":True}

class Solvent(object):

    def __init__(self):
        self.backlog = {}

    def load_puzzles(self, puzzle_dict):
        for scrambled in puzzle_dict:
            self.backlog[Path(scrambled)] = Path(puzzle_dict[scrambled])

    def load_json(self, path):
        self.truths = json.load(path)

    def load_puzzles_from_truths(self, solution_folder=None, scrambled_folder=None):
        if not(Path(solution_folder).is_dir()):
            raise ValueError("Is not a valid directory: %s"%solution_folder)
        if not(Path(scrambled_folder).is_dir()):
            raise ValueError("Is not a valid directory: %s"%scrambled_folder)

        for puzzle_path in self.truths:
            scrambled = Path(puzzle_path)
            if solution_folder is None:
                solution_folder = scrambled.parent.parent
            name = scrambled.name
            if not(scrambled_folder is None):
                scrambled = Path(scrambled_folder)/name
            reference = solution_folder/name
            self.backlog[scrambled] = reference

    def solve_one(self, scrambled, solution_im):
        open

    def detect(self, image, threshold=0.6, base_k=150, inc_k=150, sigma=0.8, *args, **kwargs):
        detector = PieceFinder(threshold = threshold)
        boxes, scores = detector.find_boxes(image, base_k, inc_k, sigma)
        subimages=list(map(lambda box: get_subarray(image, box), boxes))
        collection = PieceCollection(subimages)
        return collection

    def solve(self, collection, ref_path, pooling=5, *args, **kwargs):
        ref = openimg(ref_path)
        dims = find_dims(collection.average_shape(type="image"), len(collection), ref.shape[0:2])
        collection.dims=dims
        genpar={"generations":400, "mutate_p":0.04, "cross_p":0.13, "elitism":0.05, "selection":"tournament", "score":True}
        params=dict(pooling=pooling, debug_mode=True, iterator_mode=False,
                    id_only=False, method="genalg(xcorr)", genalg_params=genpar)
        collection, score = full_solve(collection, Solution(ref, dims), **params)
        return (collection, score)

    def load_dir(self, scrambled_dir, ref_dir, extension=".jpg", **kwargs):
        kwargs=param_check(kwargs, DEFAULTS)
        scrambled_dir=Path(scrambled_dir)
        ref_dir=Path(ref_dir)
        if not(scrambled_dir.exists()):
            raise ValueError("Directory does not exist! {!s}".format(scrambled_dir))
        if not(ref_dir.exists()):
            raise ValueError("Directory does not exist! ".format(ref_dir))
        for file in scrambled_dir.glob("*"+extension):
            try:
                if (ref_dir / file.name).exists():
                    self.backlog[file]=ref_dir / file.name
                else:
                    raise ValueError("Reference for {!s} does not exist.".format(file))
                if kwargs["debug_mode"]:
                    print("Added %s"%file.name)
            except ValueError as E:
                print(E)


    def assemble(self, collection, *args, **kwargs):
        excess = 1.1
        avg_shape = collection.average_shape()
        dims = collection.dims
        final_dims = (round(avg_shape[0]*dims[1]*excess), round(avg_shape[1]*dims[0]*excess))

        image = np.ones((final_dims[0], final_dims[1], 3), dtype = np.uint8) * 255
        locations = location_grid(avg_shape, dims, (final_dims[1]//2, final_dims[0]//2), reference="NW")
        for piece in collection.get():
            array = piece.array
            image[locations[piece.slot][1]:locations[piece.slot][1]+array.shape[0],
                  locations[piece.slot][0]:locations[piece.slot][0]+array.shape[1]] = array

        return image

    def solve_loaded(self, out_dir, log_path=None, categories=[], constants={}, *args, **kwargs):
        kwargs = param_check(kwargs, DEFAULTS)
        out_dir = Path(out_dir)
        if not(out_dir.is_dir()):
            os.mkdir(out_dir)

        if log_path is None:
            log_path = out_dir / "log.txt"
        headers = ["Scrambled_path", "Reference_path", "Total_score"] + list(constants.keys())
        appendable = list(constants.items())
        log = Logger(log_path, headers)
        start = datetime.now()
        for scrambled in self.backlog:
            try:
                if not(scrambled.exists()):
                    raise ValueError("Does not exist: %s"%scrambled)
                if not(self.backlog[scrambled].exists()):
                    raise ValueError("Does not exist: %s"%self.backlog[scrambled])

                scrambled_image = openimg(str(scrambled))
                reference_image = openimg(str(self.backlog[scrambled]))
                collection = self.detect(scrambled_image, *args, **kwargs)
                collection, score = self.solve(collection, self.backlog[scrambled], *args, **kwargs)
                out = self.assemble(collection, *args, **kwargs)
                log_dict = dict([("Scrambled_path", str(scrambled)),
                            ("Reference_path", str(self.backlog[scrambled])),
                            ("Total_score", score)] + appendable)
                #log.push_line(log_dict)

                out_path = out_dir /  scrambled.name
                print(out_path)
                if str(out_path)==str(scrambled):
                    raise ValueError("Input file must be different from output directory: "+str(out_path))
                writeimg(out_path, out)
                time=datetime.now()-start

                if kwargs["debug_mode"]:
                    print("Finished {!s}, score {:3.4}, time {}".format(scrambled,score,time))

            except Exception as E:
                #print(scrambled, ": ", E)
                raise E
        self.backlog={}
        log.close()

class Logger(object):

    def __init__(self, path, headers):
        try:
            path=Path(path)
            if not(path.suffix == ".txt"):
                raise ValueError("Log file must be .txt: "+str(path))
            self.f = open(str(path), "w")
            self.headers = []
            for header in headers:
                self.f.write("{!s} ".format(header))
                self.headers.append(header)
        except Exception as E:
            print(E)

    def push_line(self, value_dict):
        try:
            for header in self.headers:
                self.f.write("{!s} ".format(value_dict[header]))
        except Exception as E:
            print(E)

    def close(self):
        self.f.close()
        del(self)

    def __enter__(self):
        pass

    def __exit__(self):
        pass
