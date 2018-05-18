import tensorflow as tf
import cv2
import numpy as np
from glob import glob
from datetime import datetime
from timedb import DBConnector

PATTERN="*.png"
PATH=".\pieces\\"
SOLVED=".\Solved.png"
SCRAMBLEd=".\Scrambled.png"
DBPATH="data\puzzletimes.db"

def parse_img(filename, type="png"):
    image_string = tf.read_file(filename)
    if type=="png":
        image_decoded = tf.image.decode_png(image_string)
    elif type=="jpg":
        image_decoded = tf.image.decode_jpeg(image_string)
    else: raise Exception(type)
    return image_decoded

def normalize(img):
    """Normalize color vector and add 4th brightness component"""
    def norm_bright(x):
        #Using norm as brightness/(255*sqrt(3))(max value)
        if tf.norm(x)==0:
            x+=0.1
        return tf.concat([x/tf.norm(x), [tf.norm(x)/(255*3**0.5)]], axis=0)

    img=tf.cast(img, tf.float32)
    norm_img=apply_to_pixels(lambda x: norm_bright(x), img)
    return norm_img

def compare(imga,imgb):
    """Compare two non-normalized rgb images"""

    imgstack=tf.concat([imga,imgb], axis=2)

    def compare_pixel(pixela, pixelb):
        #Concatenate later
        #Compare direction using dot product, brightness using absolute difference
        #both comparisons give results in the range [0:1], with 1 as identical
        direction=tf.tensordot(pixela[0:3], pixelb[0:3], 1)
        brightness=1-tf.abs(pixela[3]-pixelb[3])
        return direction*brightness

    sum=full_sum(apply_to_pixels(lambda x:compare_pixel(x[0:4],x[4:8]), imgstack))
    return sum/tf.cast(tf.shape(imgstack)[0]*tf.shape(imgstack)[1], tf.float32)

def apply_to_pixels(function, *args):
    assert args[0].shape.ndims==3
    return tf.map_fn(lambda line: tf.map_fn(function, line), args[0])

def pool(dataset, pooling):
    return dataset.batch(1).map(lambda x:
                            tf.nn.max_pool(x, [1,pooling,pooling,1],[1,pooling,pooling,1],
                            "VALID")).map(lambda x: tf.squeeze(x, axis=0))

def full_sum(tensor):
    if tensor.shape.ndims==0:
        return tensor
    else:
        tensor=tf.foldl(lambda a,x: a+x, tensor)
        return full_sum(tensor)

def dataset_from_img_split(image, dims):
    assert type(dims)==tuple
    pieces=[]
    height=tf.shape(image)[0]/dims[0]
    width=tf.shape(image)[1]/dims[1]
    for y_split in range(dims[0]):
        for x_split in range(dims[1]):
            x_start=tf.cast(tf.floor(x_split*width), tf.int32)
            x_end=x_start+tf.cast(tf.floor(width), tf.int32)
            y_start=tf.cast(tf.floor(y_split*height), tf.int32)
            y_end=y_start+tf.cast(tf.floor(height), tf.int32)

            pieces.append(image[y_start: y_end, x_start: x_end])

    return tf.data.Dataset.from_tensor_slices(pieces)

def dataset_from_solution(path, dims):
    dataset=dataset_from_img_split(parse_img(path, path.split(".")[-1]), (dims[0],dims[1]))
    locs=[(a,b) for a in range(dims[1]) for b in range(dims[0])]
    dataset=tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(locs)))
    return dataset

def img_write(image, name):
    tf.write_file(name+".png",tf.image.encode_png(image))

def main():
    pass


if __name__ == '__main__':
    main()
