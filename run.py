from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui_proto import *
from image_obj import *

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
    """THE SOLVE FUNCTION SHOULD RETURN ID-SLOT PAIRS THAT ARE THEN SENT TO THE CANVAS OBJECT FOR A RELOCATION OR REPLOTTING
    THE CANVAS OBJECT SHOULD FIND THE NEXT LOCATION FROM EACH PIECE FROM EITHER AN INTERNAL ARRAY (NOT FROM THE COLLECTION), AND MOVE
    EACH PIECE TO ITS RESPECTIVE LOCATION, CORRECTING LOCATION AND SLOT VALUES"""
    print(1)
    id_slots = full_solve(pieces, Solution(path, pieces.dims),
                        pooling=pooling, debug_mode=True, iterator_mode=False, id_only=True, method=method)
    return id_slots

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}

root=GUI(functions)
