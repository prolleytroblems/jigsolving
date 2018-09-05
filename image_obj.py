from numba import cuda
import numpy as np
from img_recog_proto import img_split


class Solution(object):
    def __init__(self, path_or_pieces, dims):
        if isinstance(path_or_pieces, np.ndarray) and len(path_or_pieces.shape)==4:
            self.pieces=path_or_pieces
        elif isinstance(path_or_pieces, str):
            self.pieces = np.array(img_split(path_or_pieces, dims))
        else:
            try:
                 self.pieces=np.array(pieces)
                 if self.pieces.shape!=4: raise Exception
            except Exception as e:
                raise TypeError("path_or_pieces must be a path, or 4D ndarray-like of piece pbjects")
        self.dpieces=cuda.to_device(np.ascontiguousarray(self.pieces))
        self.locations=np.array([(i,j) for i in range(dims[0]) for j in range(dims[1])])
        self.availability=[True]*dims[0]*dims[1]
        self.shape=dims

class Piece(object):

    def __init__(self, image, id, location=None):
        if not(isinstance(image, np.ndarray)): raise TypeError("image must be an array")
        self.array=image
        self.id=id
        self.location=location

    def __get__():
        return self.array

class PieceCollection:

    def __init__(self, images, dims):
        self._images=images
        self.dims=dims

    def get(self):
        return self._images

    def add(self, image):
        self._images.insert(0, image)
