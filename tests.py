from img_recog_numba import *
from img_recog_proto import *
from img_recog_proto import distort as protodistort
from puzzle_gui import *

pool= 4
dims=(3,3)
brightness_delta=30

def open(path):
    return openimg(path)

def distort(images, delta):
    return [protodistort(image, 10, "b_distort") for image in images]

def split_shuffle(images, dims):
    return shuffle(images, dims)

def solve(path, pieces, dims, pooling=5):
    end=sort_pieces(locate_pieces(np.array(pieces), Solution(path, dims)), dims)
    return end

functions={"shuffle":split_shuffle, "solve":solve, "open":open, "distort":distort}
root=GUI(functions)
