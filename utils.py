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
