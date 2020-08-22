import numpy as np
"""Some simple analysis functions for the detected images"""

def color_avg(reference, channel):
    return np.average(reference[:,:,channel])

def color_stdev(reference, channel):
    return np.std(reference[:,:,channel])

def shape_avg(pieces, axis):
    return np.average(list(map(lambda x: x.shape, pieces)), axis=0)[axis]

def shape_stdev(pieces, axis):
    return np.average(list(map(lambda x: x.shape, pieces)), axis=0)[axis]

def area_avg(pieces):
    return np.average(list(map(lambda x: x.shape[0]*x.shape[1], pieces)))

def area_stdev(pieces):
    return np.std(list(map(lambda x: x.shape[0]*x.shape[1], pieces)))
