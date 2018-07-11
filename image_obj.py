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
