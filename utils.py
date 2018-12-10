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


def resize(images, ratio):
    if not(isinstance(images, list)):
        return cv2.resize(images, None, fx=ratio, fy=ratio), ratio
    else:
        return [cv2.resize(image, None, fx=ratio, fy=ratio) for image in images], ratio


def fit_to_size(images, dims, size):

    def fit_to_shape(shape, dims, size):
        if shape[1]/shape[0]>=1:
            ratio = size[0] / dims[1] / shape[1]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        elif shape[1]/shape[0]<1:
            ratio = size[1] / dims[0] / shape[0]
            new_shape = ( int( ratio * shape[1] ), int( ratio * shape[0] ) )
        return new_shape, ratio

    if isinstance(images, np.ndarray):
        shape=images.shape
        new_shape, ratio = fit_to_shape(shape, dims, size)
        return (cv2.resize(images, new_shape), ratio)

    elif isinstance(images, list):
        shape=images[0].shape
        new_shape, ratio = fit_to_shape(shape, dims, size)
        resized=[]
        for image in images:
            resized.append(cv2.resize(image, new_shape))
        return (resized, ratio)

    else:
        raise TypeError("Images must be an ndarray or list of ndarrays")


def extract_boxes(image, boxes):
    subimages=[]
    for box in boxes:
        subimages.append(image[box[1]: box[1]+box[3], box[0]: box[0]+box[2]])

    return subimages


def find_plot_locations(shape, dims, center=(400,300), reference="center"):
    if reference=="center":
        full_size=np.array((shape[1]*dims[1], shape[0]*dims[0]))
        centers=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        centers+=center-full_size//2+(shape[1]//2,shape[0]//2)
        return centers

    else: raise NotImplementedError()
