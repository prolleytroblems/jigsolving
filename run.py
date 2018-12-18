from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui_chain import *
from puzzle_gui_proto import *
from image_obj import *
from piecefinder import PieceFinder
from utils import *
import cv2

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def split_shuffle(collection, dims):
    return collection.shuffle_collection(dims)

def distort(collection, delta, mode):
    return collection.distort_collection(delta, mode)


def solve(path, pieces, pooling=5, method="xcorr", **params):
    id_slots = full_solve(pieces, Solution(path, pieces.dims),
                        pooling=pooling, debug_mode=True, iterator_mode=False, id_only=True, method=method)
    return id_slots

def detect(collection):
    detector=PieceFinder()
    image=collection.get()[0].array
    boxes, scores = detector.find_boxes(image)
    return boxes

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}
#functions={"detect":detect, "solve":solve, "open":open}

root=GUI(functions)
