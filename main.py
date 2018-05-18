

from ppiece import *
from puzzleSolver import *
import cv2
import numpy as np

def separator_func(img, **kwargs):
    pieces=[]
    for row in range(3):

        for column in range(3):
            pieces.append(img[3*row:3*row+3,3*column:3*column+3])
    return pieces

class locator:
    def __init__(self):
        self.f=self.loc()

    def loc(self):
        list=[(1,1), (2,0), (0,2), (2,1), (0,1), (1,0), (2,2), (1,2), (0,0)]
        for i in list:
            yield i

    def __call__(self, piece, img, dim, **kwargs):
        return next(self.f)
locator_func=locator()

def arranger_func(pieces, **kwargs):
    piece_array=np.zeros((3,3),dtype=object)
    for piece in pieces:
        row, column= piece.position
        piece_array[row,column]=piece.img
    last=np.vstack(np.hstack(row) for row in piece_array)

    return np.vstack(np.hstack(row) for row in piece_array)

functions=[locator_func, separator_func, arranger_func]

s=Solver("Scrambled.png", "Solved.png", functions)

s.solve()
