from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui import *

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def distort(pieces, delta, mode):
    return [protodistort(image, 10, mode) for image in pieces.get()]

def split_shuffle(pieces, dims):
    return shuffle(pieces, dims)

def solve(path, pieces, pooling=5, ids=None, **params):
    solved=full_solve(np.array(pieces.get()), Solution(path, pieces.dims), pooling=pooling, ids=ids, debug_mode=True, iterator=True)
    return solved

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}

root=GUI(functions)
