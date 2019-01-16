from utils import *
from img_recog_numba import full_solve
from piecefinder import PieceFinder
import json
import cv2
import os
from pathlib import Path
from datetime import datetime

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

    def detect(self, image, threshold, base_k, inc_k, sigma, *args, **kwargs):
        detector = PieceFinder(threshold)
        boxes, scores = detector.find_boxes(image, base_k, inc_k, sigma)
        subimages=list(map(lambda box: image.get_subimage(box), boxes))
        collection = PieceCollection(subimages)
        return collection

    def solve(self, collection, ref_path, pooling=5, *args, **kwargs):
        ref = openimg(path)
        dims = find_dims(collection.average_shape(), len(collection), ref.shape[0:2])
        collection.dims=dims
        collection, score = full_solve(collection, Solution(ref, dims),
                            pooling=pooling, debug_mode=False, iterator_mode=False,
                            id_only=False, method="genalg", score=True)
        return (collection, score)

    def assemble(self, collection, *args, **kwargs):
        excess = 1.1
        avg_shape = collection.average_shape()
        dims = collection.dims
        final_dims = (int(avg_shape[0]*dims[1]*excess), int(avg_shape[1]*dims[0]*excess))

        image = np.ones(final_dims, dtype = np.uint8) * 255
        locations = location_grid(avg_shape, dims, final_dims//2)

        for piece in collection.pieces:
            image[locations[piece.location][1]:avg_shape[0],
                  locations[piece.location][0]:avg_shape[1]] = piece.array

        return image

    def solve_loaded(self, out_dir, log_path=None, categories=[], constants={}, *args, **kwargs):
        out_dir = Path(out_dir)
        if not(out_dir.is_dir()):
            os.mkdir(out_dir)

        if log_path is none:
            log_path = out_dir / "log.txt"
        headers = ["Scrambled_path", "Reference_path", "Total_score"] + list(constants.keys())
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
                collection = self.detect(scrambled_image, eyyyy)
                collection, score = self.solve(collection, self.backlog["scrambled"], *args, **kwargs)
                out = self.assemble(collection, *args, **kwargs)
                log_dict = {"Scrambled_path": str(scrambled),
                            "Reference_path": str(self.backlog[scrambled]),
                            "Total_score": score} + constants
                log.push_line()
                del(self.backlock[scrambled])
                out_path = out_dir /  scrambled.name
                if str(out_path)==str(scrambled):
                    raise ValueError("Input file must be different from output directory: "+str(out_path))
                writeimg(out_path, out)
                time=start-datetime.now()
                print(time, scrambled)
            except E:
                print(scrambled, ": ", E)
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
        except E:
            print(E)

    def push_line(self, value_dict):
        try:
            for header in self.headers:
                self.f.write("{!s} ".format(value_dict[header]))
        except E:
            print(E)

    def close(self):
        self.f.close()
        del(self)

    def __enter__(self):
        pass

    def __exit__(self):
        pass
