from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui_chain import *
#from puzzle_gui_proto import *
from image_obj import *
from piecefinder import PieceFinder
from utils import *
import cv2
from dataset import *
from pathlib import Path

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def split_shuffle(collection, dims):
    return collection.shuffle_collection(dims)

def distort(collection, delta, mode):
    return collection.distort_collection(delta, mode)


def solve(path, collection, pooling=5, method="xcorr", **params):
    solutionimg=cv2.imread(path)
    print(len(collection))
    dims = find_dims(collection.average_shape(), len(collection), solutionimg.shape[0:2])
    collection.dims=dims
    print(dims)

    id_slots = full_solve(collection, Solution(solutionimg, dims),
                        pooling=pooling, debug_mode=True, iterator_mode=False, id_only=True, method=method)
    return id_slots

def detect(collection, threshold):
    detector=PieceFinder(threshold=threshold)
    image=collection.get()[0].array
    boxes, scores = detector.find_boxes(image)
    return boxes

def show(path):
    image=cv2.imread(path)
    cv2.imshow("Solution image", image)
    cv2.waitKey(1)

genner=ImageSplitter()
genner.gen(Path("./images/puzzle.jpg"), Path("./images/samples/"), dims=(4,5), min =-1)
genner.close()

#functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}
functions={"detect":detect, "solve":solve, "open":open, "distort":distort, "show":show}

root=GUI(functions)
