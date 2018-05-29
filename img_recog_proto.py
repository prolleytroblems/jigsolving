import numpy as np
import cv2
from random import normalvariate as nrand
from numpy.linalg import norm

def openimg(path):
    return bgr_to_rgb(cv2.imread(path, 1))

def writeimg(name, image):
    cv2.imwrite(name, rgb_to_bgr(image))

def rgb_to_bgr(image):
    r,g,b=np.split(image, 3, axis=2)
    return np.concatenate((b,g,r), axis=2)

def bgr_to_rgb(image):
    b,g,r=np.split(image, 3, axis=2)
    return np.concatenate((r,g,b), axis=2)

def img_split_cpu(image_or_path, dims):
    if type(image_or_path)==str:
        image=openimg(image_or_path)
    else:
        image=image_or_path
    assert type(dims)==tuple

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

def shuffle(image, dims):
    """Shuffle the image into equal rectangular pieces"""
    img_split_cpu(image, dims)


def b_distort(image, delta):
    """Randomly alter the brightness of each pixel of an image following a normal distribution."""

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
    return np.apply_along_axis(func, 2, image)


def s_distort(image, delta):
    """Randomly alter the shape of an image"""

def ub_distribution(image, delta, fixed_points):
    """Randomly alter the brightness of an image as a whole."""


def c_distort(image, delta):
    """Randomly alter the color vector of each pixel of an image following a normal distribution."""

def ub_distribution(image, delta, fixed_points):
    """Randomly alter the color of an image as a whole."""

writeimg("distorted.jpg", b_distort(openimg("puzzle.jpg"), 10))
