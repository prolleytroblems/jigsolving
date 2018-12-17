from numba import cuda
import numpy as np
from img_recog_proto import img_split, shuffle, distort
from utils import find_plot_locations, get_subarray


class Solution(object):
    def __init__(self, path_or_arrays, dims):
        if isinstance(path_or_arrays, np.ndarray) and len(path_or_arrays.shape)==4:
            self.arrays=path_or_arrays
        elif isinstance(path_or_arrays, str):
            self.arrays = np.array(img_split(path_or_arrays, dims))
        else:
            try:
                 self.arrays=np.array(arrays)
                 if self.arrays.shape!=4: raise Exception
            except Exception as e:
                raise TypeError("path_or_arrays must be a path, or 4D ndarray-like of array pbjects")
        self.darrays=cuda.to_device(np.ascontiguousarray(self.arrays))
        self.slots=[(i,j) for i in range(dims[0]) for j in range(dims[1])]
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

    def get_subimage(self, box):
        return get_subarray(self.array, box)


class PieceCollection:

    def __init__(self, pieces_or_images, dims):
        if isinstance(pieces_or_images[0], np.ndarray):
            images=pieces_or_images
            if len(images[0].shape)==3:
                assert dims[0]*dims[1]==len(images)
            elif len(np.array(images).shape)==3:
                assert dims[0]*dims[1]==1
                images=[images]
            else:
                raise Exception("Invalid image object")
            self._pieces=[Piece(image) for image in images]
            self.dims=dims
        elif isinstance(pieces_or_images[0], Piece):
            self._pieces=pieces_or_images
            self.dims=dims
        self.slot_dict={}
        self.id_dict={}
        self._invalid_slots=False

    def get(self, slot=None, id=None):
        if not(slot is None):
            try:
                piece = self.slot_dict[slot]
            except:
                piece = self.slot_dict[tuple(slot.tolist())]
            return piece
        elif not(id is None):
            return self.id_dict[id]
        else:
            return self._pieces

    def sort(self):
        new=[]
        for y in range(self.dims[0]):
            for x in range(self.dims[1]):
                new.append(self.slot_dict[(y,x)])
        self._pieces=new

    def add(self, image):
        self._pieces.insert(0, Piece(image))

    def mass_set(self, attr, values):
        assert len(values)==len(self._pieces)
        if attr=="id":
            for value, piece in zip(values, self._pieces):
                piece.id=value
                self.id_dict[value]=piece
        elif attr=="plotted":
            for value, piece in zip(values, self._pieces):
                piece.plotted=value
        elif attr=="location":
            for value, piece in zip(values, self._pieces):
                piece.location = value
            self._invalid_slots = True
        elif attr=="tkimage":
            for value, piece in zip(values, self._pieces):
                piece.tkimage=value
        elif attr=="slot":
            for value, piece in zip(values, self._pieces):
                piece.slot=value
                self.slot_dict[value]=piece
            self.sort()
        else:
            raise Exception("This attribute does not exist.")

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
            if self._invalid_slots:
                raise Exception()
            return([piece.slot for piece in self._pieces])
        else:
            raise Exception("This attribute does not exist.")

    def shuffle_collection(self, dims):
        images=np.array(self.mass_get("image"))
        images=shuffle(images, dims, self.dims)
        return PieceCollection(images, dims)

    def distort_collection(self, delta, mode):
        for piece in self._pieces:
            piece.array = distort(piece.array, delta, mode)
        return self

    def __len__(self):
        return len(self._pieces)
