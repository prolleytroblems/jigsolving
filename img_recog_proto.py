import numpy as np
from random import normalvariate as nrand
from random import sample, random
from numba import cuda, guvectorize
import math
from utils import *


def img_split(image_or_path, dims):
    "Splits an image into rectangular, equally-sized pieces. Returns a list, not an ndarray."
    if isinstance(image_or_path, str):
        image=openimg(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        if len(image_or_path.shape)==3:
            image=image_or_path
        else:
            raise TypeError("image_or_path must be of type str or np.ndarray")
    else:
        raise TypeError("image_or_path must be of type str or np.ndarray")

    assert isinstance(dims, tuple)

    pieces=[]
    height=image.shape[0]/dims[0]
    width=image.shape[1]/dims[1]
    for y_split in range(dims[0]):
        for x_split in range(dims[1]):
            x_start=int(x_split*width)
            x_end=x_start+int(width)
            y_start=int(y_split*height)
            y_end=y_start+int(height)

            pieces.append(np.array(image[y_start: y_end, x_start: x_end]))
    return pieces


def shuffle(images, dims):
    """Shuffle the image into equal rectangular pieces"""
    images=np.array(images)
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple.")
    if len(images.shape)==4:
        if images.shape[0]==1:
            return sample(img_split(images[0], dims), dims[0]*dims[1])
        else:
            return shuffle(reassemble(images, dims), dims)
    elif len(images.shape)==3:
        return sample(img_split(images, dims), dims[0]*dims[1])
    else:
        raise TypeError("Array is not legible as image.")
    return


def reassemble(pieces, dims):
    """Reassembles ordered piece images into a full image"""
    pieces=np.array(pieces)
    if not(len(pieces.shape)==4): raise TypeError("pieces must be a 4-dimensional ndarray-like object")
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple")
    image=np.concatenate([np.concatenate(pieces[i*dims[1]:(i+1)*dims[1]], axis=1) for i in range(dims[0])], axis=0)
    return image


def distort(image, delta, distortion):
    if not(len(np.array(image.shape))==3): raise TypeError("Array is not legible as image")
    if distortion=="n":
        return b_distort(image, delta)
    if distortion=="s":
        return s_distort(image, delta)
    else:
        raise Exception("Not implemented!")


@guvectorize("(uint8[:], float32, uint8[:])","(m),()->(m)")
def b_distort(pixel, delta, res):
    change=int(nrand(0, delta))
    for i in range(3):
        value=pixel[i]+change
        if value>255:
            res[i]=255
        elif value<0:
            res[i]=0
        else:
            res[i]=value


def s_distort(image, delta):
    """Randomly alter the shape of an image"""
    def move_one(image, axis, side="end"):
        assert axis==0 or axis==1
        #add a blank row/column
        coeff=[0,0]
        coeff[-axis+1]=1
        if side=="end":
            image=np.concatenate((image, np.zeros(((image.shape[0]-1)*coeff[0]+1,
                    (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8)), axis=axis)
        elif side=="start":
            image=np.concatenate((np.zeros(((image.shape[0]-1)*coeff[0]+1,
                    (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8), image), axis=axis)
        return image

    direction=random()*math.pi*2
    tx=int(math.cos(direction)*delta)
    ty=int(math.sin(direction)*delta)
    for axis, amount in enumerate([ty,tx]):
        if amount <=0:
            side="end"
        else:
            side="start"
        for i in range(abs(amount)):
            image=move_one(image, axis, side)


    return image


def ub_distribution(image, delta, fixed_points):
    """Randomly alter the brightness of an image as a whole."""
    pass


def c_distort(image, delta):
    """Randomly alter the color vector of each pixel of an image following a normal distribution."""
    pass


def ub_distribution(image, delta, fixed_points):
    """Randomly alter the color of an image as a whole."""
    pass


if __name__=="__main__":
    pass
