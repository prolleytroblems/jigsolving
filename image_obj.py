from numba import cuda
import numpy as np
from img_recog_proto import img_split, shuffle
import cv2


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

    def __init__(self, image, id=None, location=None):
        if not(isinstance(image, np.ndarray)): raise TypeError("image must be an array")
        self.array=image
        self.id=id
        self.location=location
        self.plotted=None
        self.tkimage=None
        self.slot=None

    def __get__():
        return self.array


class PieceCollection:

    def __init__(self, images, dims):
        if len(np.array(images).shape)==4:
            assert dims[0]*dims[1]==len(images)
        elif len(np.array(images).shape)==3:
            assert dims[0]*dims[1]==1
            images=[images]
        else:
            raise Exception("Invalid image object")
        self._pieces=[Piece(image) for image in images]
        self.dims=dims
        self.ph_dict={}

    def get(self, location=None):
        if location == None:
            return self._pieces
        else:
            assert self.loc_dict

    def add(self, image):
        self._pieces.insert(0, image)

    def mass_set(self, attr, values):
        assert len(values)==len(self._pieces)
        if attr=="id":
            for value, piece in zip(values, self._pieces):
                piece.id=value
        elif attr=="plotted":
            for value, piece in zip(values, self._pieces):
                piece.plotted=value
        elif attr=="location":
            for value, piece in zip(values, self._pieces):
                piece.location = value
        elif attr=="tkimage":
            for value, piece in zip(values, self._pieces):
                piece.tkimage=value
        elif attr=="slot":
            for value, piece in zip(values, self._pieces):
                piece.slot=value
                self.ph_dict[value]=piece

    def mass_get(self, attr):
        if attr=="id":
            return([piece.id for piece in self._pieces])
        elif attr=="plotted":
            return([piece.plotted for piece in self._pieces])
        elif attr=="location":
            return([piece.location for piece in self._pieces])
        elif attr=="tkimage":
            return([piece.tkimage for piece in self._pieces])
        elif attr=="image":
            return([piece.array for piece in self._pieces])
        elif attr=="slot":
            return([piece.slot for piece in self._pieces])

    @staticmethod
    def shuffle_collection(collection, dims):
        images=np.array(collection.mass_get("image"))
        images=shuffle(images, dims, collection.dims)
        return PieceCollection(images, dims)
