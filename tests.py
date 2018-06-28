from img_recog_numba import *
from img_recog_proto import *
from puzzle_gui import *

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def distort(images, delta):
    return [b_distort(image, 10) for image in images]

def split_shuffle(images, dims):
    return shuffle(images[0], dims)

def solve(path, pieces, dims, pooling=5):
    end=sort_pieces(locate_pieces(np.array(pieces), Solution(path, dims)), dims)
    return end

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}
root=GUI(functions)
