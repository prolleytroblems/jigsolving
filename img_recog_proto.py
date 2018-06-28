import numpy as np
from numpy.linalg import norm
import cv2
from random import normalvariate as nrand
from random import sample
from PIL import Image
from numba import cuda
from pathlib import Path

def openimg(filepath):
    def bgr_to_rgb(image):
        b,g,r=np.split(image, 3, axis=2)
        return np.concatenate((r,g,b), axis=2)

    try:
        mypath=Path(filepath)
        if not(mypath.is_file()):
            raise IOError("That file doesn't exist!")
        return bgr_to_rgb(cv2.imread(filepath, 1))
    except IOError as e:
        print(e)

def writeimg(name, image):
    def rgb_to_bgr(image):
        r,g,b=np.split(image, 3, axis=2)
        return np.concatenate((b,g,r), axis=2)
    try:
        if not(isinstance(image, np.adarray) and len(image.shape)==4): raise TypeError("File is illegible as image!")
        cv2.imwrite(name, rgb_to_bgr(image))
    except TypeError as e:
        print(e)


def img_split_cpu(image_or_path, dims):
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
            return sample(img_split_cpu(images[0], dims), dims[0]*dims[1])
        else:
            return shuffle(reassemble(images, dims), dims)
    elif len(images.shape)==3:
        return sample(img_split_cpu(images, dims), dims[0]*dims[1])
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

def b_distort(image, delta):
    """Randomly alter the brightness of each pixel of an image following a normal distribution."""
    if not(len(np.array(image.shape))==3): raise TypeError("Array is not legible as image")
    def func(pixel_values):
        change=int(nrand(0, delta))

        new_values=[]
        for value in pixel_values.astype(int)+change:
            if value>255:
                new_values.append(255)
            elif value<0:
                new_values.append(0)
            else:
                new_values.append(value)
        return np.array(new_values)
    return np.apply_along_axis(func, 2, image).astype("uint8")


def s_distort(image, delta):
    """Randomly alter the shape of an image"""
    pass

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
