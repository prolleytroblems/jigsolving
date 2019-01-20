from utils import *
from img_recog_numba import full_solve
from piecefinder import PieceFinder
import json
import cv2
import os
from pathlib import Path
from datetime import datetime
from image_obj import PieceCollection, Solution
from piecefinder import FindDimsFailure

DEFAULTS = {"debug_mode":True}

class Solvent(object):

    def __init__(self):
        self.backlog = {}

    def config(self, processing_functions):
        self.processing_functions = processing_functions

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

    def detect(self, image, threshold=0.8, base_k=150, inc_k=150, sigma=0.8, ref_shape=None, *args, **kwargs):
        detector = PieceFinder(threshold = threshold, max_loss=0.01, max_tries=10)
        boxes, scores, dims = detector.find_boxes(image, base_k, inc_k, sigma, ref_shape=ref_shape)
        subimages = list(map(lambda box: get_subarray(image, box), boxes))
        if len(boxes) ==0:
            raise RuntimeWarning("Could not find any pieces.")
        collection = PieceCollection(subimages)
        return collection, dims

    def solve(self, collection, ref_image, dims=None, pooling=5, *args, **kwargs):
        if dims is None:
            dims, loss = find_dims(collection.average_shape(type="image"), len(collection), ref_image.shape[0:2])
        collection.dims=dims
        genpar={"generations":400, "mutate_p":0.09, "cross_p":0.13, "elitism":0.05, "selection":"tournament", "score":True}
        for key in genpar:
            if key in kwargs:
                genpar[key] = kwargs[key]
        params=dict(pooling=pooling, debug_mode=True, iterator_mode=False,
                    id_only=False, method="genalg(xcorr)", genalg_params=genpar)
        collection, score = full_solve(collection, Solution(ref_image, dims), **params)
        return (collection, score)

    def assemble(self, collection, *args, **kwargs):
        excess = 1.5
        avg_shape = collection.average_shape()
        dims = collection.dims
        final_dims = (round(avg_shape[0]*dims[0]*excess), round(avg_shape[1]*dims[1]*excess))
        image = np.ones((final_dims[0], final_dims[1], 3), dtype = np.uint8) * 255
        locations = location_grid(avg_shape, dims, (final_dims[1]//2, final_dims[0]//2), reference="NW")
        try:
            for piece in collection.get():
                array = piece.array
                image[locations[piece.slot][1]:locations[piece.slot][1]+array.shape[0],
                      locations[piece.slot][0]:locations[piece.slot][0]+array.shape[1]] = array
        except Exception as E:
            print("dims", dims)
            print("avg_shape", avg_shape)
            print("final_dims", image.shape)
            print("array shape", array.shape)
            raise E
        return image

    def solve_loaded(self, out_dir, log_path=None, constants={}, *args, **kwargs):
        kwargs = param_check(kwargs, DEFAULTS)
        out_dir = Path(out_dir)
        if not(out_dir.is_dir()):
            os.mkdir(out_dir)

        if log_path is None:
            log_path = out_dir / "log.txt"
        headers = ["Scrambled_path", "Reference_path", "Total_score", "Exception"] + list(constants.keys()) + list(self.processing_functions.keys())
        appendable = list(constants.items())
        with Logger(log_path, headers) as log:
            start = datetime.now()
            for scrambled in self.backlog:
                try:
                    data = {}
                    score = None

                    if not(scrambled.exists()):
                        raise ValueError("Does not exist: %s"%scrambled)
                    if not(self.backlog[scrambled].exists()):
                        raise ValueError("Does not exist: %s"%self.backlog[scrambled])

                    scrambled_image = openimg(str(scrambled))
                    reference_image = openimg(str(self.backlog[scrambled]))
                    collection, dims = self.detect(scrambled_image, ref_shape=reference_image.shape[0:2], **kwargs)

                    piece_count = len(collection)

                    for key in self.processing_functions:
                        data[key] = self.processing_functions[key](collection.mass_get("image"), reference_image)

                    collection, score = self.solve(collection, reference_image, dims=dims, **kwargs)
                    out = self.assemble(collection, *args, **kwargs)


                    out_path = out_dir /  scrambled.name
                    if str(out_path)==str(scrambled):
                        raise ValueError("Input file must be different from output directory: "+str(out_path))
                    writeimg(out_path, out)
                    time=datetime.now()-start

                    if kwargs["debug_mode"]:
                        print("Finished {!s}, score {:3.4}, time {}".format(scrambled,score,time))

                except (ValueError, FindDimsFailure, RuntimeWarning) as E:
                    print(scrambled, ": ", E)
                    data["Exception"] = str(E)
                    if score is None:
                        score = 0

                finally:
                    log_dict = dict([("Scrambled_path", str(scrambled)),
                                ("Reference_path", str(self.backlog[scrambled])),
                                ("Total_score", score)] + list(constants.items()) + list(data.items()))
                    log.push_line(log_dict)

        self.backlog={}
        log.close()

class Logger(object):

    def __init__(self, path, headers):
        self.path=Path(path)
        if not(path.suffix == ".txt"):
            raise ValueError("Log file must be .txt: "+str(path))
        self.headers = headers

    def push_line(self, value_dict):
        for header in self.headers:
            if header in value_dict:
                self.f.write("{!s}, ".format(value_dict[header]))
            else:
                self.f.write(", ")
        self.f.write("\n")

    def open(self):
        self.f = open(self.path, "w")
        for header in self.headers:
            self.f.write("{!s}, ".format(header))
        self.f.write("\n")

    def close(self):
        self.f.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
