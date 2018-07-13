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

def img_split(image_or_path, dims):
    "Splits an image into rectangular, equally-sized pieces. Returns a list, not an ndarray."
    if isinstance(image_or_path, str):
        image=openimg(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        if len(image_or_path.shape)==3:
            image=image_or_path
        else:
            raise TypeError("image_or_path must be of 3 dimensions ("+str(len(image_or_path.shape))+" dimensions)")
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


def reassemble(pieces, dims):
    """Reassembles ordered piece images into a full image"""
    if len(pieces)==1:
        return np.array(pieces[0])

    pieces = np.array(pieces)
    if not(len(pieces.shape)==4): raise TypeError("pieces must be a 4-dimensional ndarray-like object")
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple")
    image = np.concatenate([np.concatenate(pieces[i*dims[1]:(i+1)*dims[1]], axis=1) for i in range(dims[0])], axis=0)

    return image
