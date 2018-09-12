from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui import *
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


def solve(path, pieces, pooling=5, ids=None, **params):

    THE SOLVE FUNCTION SHOULD RETURN ID-SLOT PAIRS THAT ARE THEN SENT TO THE CANVAS OBJECT FOR A RELOCATION OR REPLOTTING
    THE CANVAS OBJECT SHOULD FIND THE NEXT LOCATION FROM EACH PIECE FROM EITHER AN INTERNAL ARRAY (NOT FROM THE COLLECTION), AND MOVE
    EACH PIECE TO ITS RESPECTIVE LOCATION, CORRECTING LOCATION AND SLOT VALUES

    solved=full_solve(np.array(pieces.get()), Solution(path, pieces.dims), pooling=pooling, ids=ids, debug_mode=True, iterator=True)
    return solved

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}

root=GUI(functions)
