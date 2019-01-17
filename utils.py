from pathlib import Path
import cv2
import numpy as np
from copy import copy

def get_subarray(array, box, expansion=0):
    box=(box[0]-expansion, box[1]-expansion, box[2]+2*expansion, box[3]+2*expansion)
    if box[0]<0:
        padding = np.ones((box[3], -box[0], 3))*255
        subbox=(0, box[1], box[2]+box[0], box[3])
        subarray = get_subarray(array, subbox, expansion=0)
        return np.concatenate((padding, subarray), axis=1)
    elif box[0]+box[2]>array.shape[1]:
        padding = np.ones((box[3], box[0]+box[2]-array.shape[1], 3))*255
        subbox=(box[0], box[1], array.shape[1]-box[0], box[3])
        subarray = get_subarray(array, subbox, expansion=0)
        return np.concatenate((subarray, padding), axis=1)
    if box[1]<0:
        padding = np.ones((-box[1], box[2], 3))*255
        subbox=(box[0], 0, box[2], box[3]+box[1])
        subarray = get_subarray(array, subbox, expansion=0)
        return np.concatenate((padding, subarray), axis=0)
    elif box[1]+box[3]>array.shape[0]:
        padding = np.ones((box[1]+box[3]-array.shape[0], box[2], 3))*255
        subbox=(box[0], box[1], box[2], array.shape[0]-box[1])
        subarray = get_subarray(array, subbox, expansion=0)
        return np.concatenate((subarray, padding), axis=0)

    corners=(box[1], box[1]+box[3],
             box[0], box[0]+box[2])

    return array[corners[0]:corners[1], corners[2]:corners[3]]

def img_split(image_or_path, dims, invert=True):
    "Splits an image into rectangular, equally-sized pieces. Returns a list, not an ndarray."
    if isinstance(image_or_path, str):
        if invert:
            image=openimg(image_or_path)
        else:
            image=cv2.imread(image_or_path)
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

def param_check(params, defaults):
    """Tool for filling out parameter list with defaults."""
    if not(isinstance(params, dict) and isinstance(defaults, dict)): raise TypeError("Both inputs must be dictionaries.")

    for key in defaults:
        if not(key in params):
            params[key]=defaults[key]

    return params

def apply_transformation(array, transf_matrix):
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            array[y,x,:]=np.matmul(traf_matrix,array[y,x,:])
    return array

def reflect(array):
    """Flip x and y axes"""
    flipped=np.empty((array.shape[1], array.shape[0], array.shape[2]), dtype=np.uint8)
    for row in range(flipped.shape[0]):
        flipped[row]=array[:,row]
    return flipped

def openimg(filepath):
    """Wrapper for cv2.imread to correct channel order."""
    def bgr_to_rgb(image):
        b,g,r=np.split(image, 3, axis=2)
        return np.concatenate((r,g,b), axis=2)

    try:
        filepath=Path(filepath)
        if not(filepath.is_file()):
            raise IOError("That file doesn't exist!")
        return bgr_to_rgb(cv2.imread(str(filepath), 1))
    except IOError as e:
        print(e)

def writeimg(path, image):
    """Wrapper for cv2.imwrite to correct channel order."""
    def rgb_to_bgr(image):
        r,g,b=np.split(image, 3, axis=2)
        return np.concatenate((b,g,r), axis=2)
    try:
        if not(isinstance(image, np.ndarray) and len(image.shape)==3): raise TypeError("File is illegible as image!")
        cv2.imwrite(str(Path(path)), rgb_to_bgr(image))
    except TypeError as E:
        raise(E)

def resize(images, ratio):
    if not(isinstance(images, list)):
        return cv2.resize(images, None, fx=ratio, fy=ratio), ratio
    else:
        return [cv2.resize(image, None, fx=ratio, fy=ratio) for image in images], ratio

def fit_to_size(images, dims, size):
    """Rescales list or array of equally sized image arrays to fill the maximum
        of a certain space without exceeding any dimension. Receives dims in
        array notation, size in pixel notation."""
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
    """Receives an array of at least 2D and iteratable of boxes (x0,y0,w,h), returns list of subimages"""
    subimages=[]
    for box in boxes:
        subimages.append(image[box[1]: box[1]+box[3], box[0]: box[0]+box[2]])

    return subimages

def find_plot_locations(shape, dims, center=(400,300), reference="center"):
    """Receives shapes and dims in array notation (H,W), center in pixel notation,
        returns centers in pixel notation (W,H), (X,Y)"""
    if reference=="center":
        full_size=np.array((shape[1]*dims[1], shape[0]*dims[0]))
        centers=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        centers+=np.array(center)-full_size//2+(shape[1]//2,shape[0]//2)
        return np.array(np.round(centers), dtype=np.int32)
    if reference=="NW":
        full_size=np.array((shape[1]*dims[1], shape[0]*dims[0]))
        corners=np.array([(x*shape[1], y*shape[0]) for y in range(dims[0]) for x in range(dims[1])])
        corners+=np.array(center)-full_size//2
        return np.array(np.round(corners), dtype=np.int32)

    else: raise NotImplementedError()

def location_grid(shape, dims, center=(400,300), reference="center"):
    """Receives shapes and dims in array notation (H,W), center in pixel notation,
        returns centers in pixel notation (W,H), (X,Y). Return format is a (dim[0],dims[1])-shaped array"""
    locations=find_plot_locations(shape, dims, center=center, reference=reference)
    locations=np.array(locations)
    return np.reshape(locations, (dims[0], dims[1], 2))

def get_divisors(number):
    """Return a list of all divisors of given number, ordered by pairs, in
        increasing order of the first of the two. If perfect square, last two
        numbers are the same. e.g. 12->[1,12,2,6,3,4] """
    last=None
    current=1
    running=True
    divisors=[]
    while running:
        if number%current==0:
            last=number//current
            divisors.append(current)
            divisors.append(number//current)
        current+=1
        if current>last:
            running=False
    return divisors

def find_dims(piece_shape, piece_count, full_shape):
    """Piece and reference shapes in array notation (H, W), returns dim values in array notation (rows, columns)"""
    assert isinstance(piece_count, int)
    potential_dims=np.array(get_divisors(piece_count))
    potential_dims=np.reshape(potential_dims, (potential_dims.shape[0]//2, 2))
    potential_dims=np.concatenate((potential_dims, np.zeros((potential_dims.shape[0], 1))), axis=1)
    sol_proportion = full_shape[1]/full_shape[0]
    for pair in potential_dims:
        loss = (piece_shape[1]*pair[1]/(piece_shape[0]*pair[0])-sol_proportion)**2
        pair[2] = loss
    index =  np.argmin(potential_dims[:, 2])
    out = tuple(np.array(potential_dims[index, 0:2], dtype=np.uint8).tolist())
    loss = potential_dims[index, 2]
    return out, loss

def exact_reassemble(pieces, dims):
    """Reassembles ordered piece images into a full image"""
    assert dims[0]*dims[1]==len(pieces)
    pieces=np.array(pieces)
    if not(len(pieces.shape)==4): raise TypeError("pieces must be a 4-dimensional ndarray-like object")
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple")
    image=np.concatenate([np.concatenate(pieces[i*dims[1]:(i+1)*dims[1]], axis=1) for i in range(dims[0])], axis=0)
    return image

def mili_seconds(dt_delta):
    return str(dt_delta.seconds*1000+float(dt_delta.microseconds)/1000)+" ms"
