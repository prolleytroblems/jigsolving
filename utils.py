from pathlib import Path
import cv2
import numpy as np


def param_check(params, defaults):
    if not(isinstance(params, dict) and isinstance(defaults, dict)): raise TypeError("Both inputs must be dictionaries.")

    for key in defaults:
        if not(key in params):
            params[key]=defaults[key]

    return params


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


def resize(images, dims, size):
    if isinstance(images, np.ndarray):
        shape=images.shape
        if shape[1]/shape[0]>=1:
            ratio = size[0] / dims[1] / shape[1]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        elif shape[1]/shape[0]<1:
            ratio = size[1] / dims[0] / shape[0]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        return (cv2.resize(images, new_shape), ratio)

    elif isinstance(images, list):
        shape=images[0].shape
        if shape[1]/shape[0]>=1:
            ratio = size[0] / dims[1] / shape[1]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        elif shape[1] / shape[0] < 1:
            ratio = size[1] / dims[0] / shape[0]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        resized=[]
        for image in images:
            resized.append(cv2.resize(image, new_shape))
        return (resized, ratio)
    else:
        raise TypeError("Images must be an ndarray or list of ndarrays")
