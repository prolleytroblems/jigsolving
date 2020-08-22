import cv2
import numpy as np
from math import pi, exp, sqrt, tan, radians

def motion_blur_kernel(direction=0, kernel_size=9):
    "direction in degrees, starting from positive x direction, counter-lockwise"
    direction = direction % 360
    kernel=np.zeros((kernel_size, kernel_size), dtype=np.float32)
    half = kernel_size//2+1
    if direction == 0:
        for i in range(half):
            kernel[kernel_size//2, kernel_size//2+i]=1/(half)
    elif direction == 90:
        for i in range(half):
            kernel[kernel_size//2-i, kernel_size//2]=1/(half)
    elif direction == 180:
        for i in range(half):
            kernel[kernel_size//2, kernel_size//2-i]=1/(half)
    elif direction == 270:
        for i in range(half):
            kernel[kernel_size//2+i, kernel_size//2]=1/(half)
    else:
        prop=tan(radians(direction))
        for x in range(half):
            if direction>90 and direction<270:
                x=-x
            y = kernel_size//2-round(x*prop)
            if y>=0 and y<=kernel_size-1:
                kernel[y, kernel_size//2+x]=1
        for y in range(half):
            if direction>180:
                y=-y
            x = kernel_size//2+round(y/prop)
            if x>=0 and x<=kernel_size-1:
                kernel[kernel_size//2-y, x]=1
        kernel=kernel/np.sum(kernel)
    return kernel

def motion_blur(array, *args, **kwargs):
    return cv2.filter2D(array, -1, motion_blur_kernel(*args, **kwargs))

def gaussian_blur(array, stddev=3, kernel_size=(9,9)):
    return cv2.GaussianBlur(array, ksize=kernel_size, sigmaX=stddev)

def median_blur(array, kernel_size=(3,3)):
    return cv2.medianBlur(array, ksize=kernel_size)

def normal_blur(array, kernel_size=(3,3)):
    return cv2.blur(array, ksize=kernel_size)
