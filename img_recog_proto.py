import numpy as np
from random import normalvariate as nrand
from random import sample, random
from numba import cuda, guvectorize
import math
from utils import *
from filters import *



def _raw_shuffle(image, dims):
    return sample(img_split(image, dims), dims[0]*dims[1])


def shuffle(images, dims, prev_dims):
    """Shuffle the image into equal rectangular pieces"""
    images=np.array(images)
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple.")
    if len(images.shape)==4:
        if images.shape[0]==1:
            return _raw_shuffle(images[0], dims)
        else:
            return _raw_shuffle(reassemble(images, prev_dims), dims)
    elif len(images.shape)==3:
        return _raw_shuffle(images, dims)
    else:
        raise TypeError("Array is not legible as image.")


def distort(image, delta, distortion):
    if not(len(np.array(image.shape))==3): raise TypeError("Array is not legible as image")
    if distortion=="g":
        return b_distort_r(image, np.float32(delta))
    elif distortion=="c":
        return c_distort(image, delta)
    elif distortion=="b":
        return b_distort_f(image, np.uint8(delta))
    elif distortion=="m":
        return m_distort(image, delta)
    elif distortion=="bl":
        return bl_distort(image, delta)
    else:
        raise Exception("Not implemented!")


@guvectorize("(uint8[:], float32, uint8[:])","(m),()->(m)")
def b_distort_r(pixel, delta, res):
    "Gaussian noise."
    change=int(nrand(0, delta))
    for i in range(3):
        value=pixel[i]+change
        if value>255:
            res[i]=255
        elif value<0:
            res[i]=0
        else:
            res[i]=value

@guvectorize("(uint8[:], uint8, uint8[:])","(m),()->(m)")
def b_distort_f(pixel, delta, res):
    "Fixed brightness change."
    change=int(delta)
    for i in range(3):
        value=pixel[i]+change
        if value>255:
            res[i]=255
        elif value<0:
            res[i]=0
        else:
            res[i]=value


def c_distort(image, delta):
    """Randomly slide image"""
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


def bl_distort(image, delta):
    """Gaussian blur."""
    return gaussian_blur(image, stddev=delta, kernel_size=(int(delta)*2+1, int(delta)*2+1))

def m_distort(image, delta):
    """Motion blur."""
    return motion_blur(image, direction=int(360*random()), kernel_size=int(delta)*2+1)


def c_distort(image, delta):
    """Randomly alter the color vector of each pixel of an image following a normal distribution."""
    pass


def ub_distribution(image, delta, fixed_points):
    """Randomly alter the color of an image as a whole."""
    pass


if __name__=="__main__":
    pass
