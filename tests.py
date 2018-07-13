from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui import *
from image_obj import Piece

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def distort(pieces, delta, mode):
    return [Piece(protodistort(piece.array, 10, mode), None, None) for piece in pieces]

def split_shuffle(pieces, dims):
    return shuffle_pieces(pieces, dims)

def solve(path, pieces, dims, pooling=5, ids=None, **params):
    solved=full_solve(pieces, Solution(path, dims), pooling=pooling, ids=ids, debug_mode=True, iterator=True)
    return solved

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}

root=GUI(functions)
