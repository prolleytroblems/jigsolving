"""Solves a jigsaw puzzle received as an image."""

#v0.1 Placeholder functions and objects
import cv2
import numpy as np
from ppiece import *
import tensorflow as tf

class Solver:

    def __init__(self, unsolved_path, solution_path, functions, **kwargs):
        """Receives the location of the unsolved and solved puzzle images as
        location strings"""
        self.pieces_path=unsolved_path
        self.solution_path=solution_path
        self.locate=Solver.locator(functions[0])
        self.find_pieces=Solver.finder(functions[1])
        self.arrange=Solver.arranger(functions[2])
        self.solved=None


    def solve(self, **kwargs):

        piece_dataset=self.find_pieces(self.pieces_path)
        piece_dataset=locate(piece_dataset, self.solution_path)
        img=Solver.parse_img(solution_path, **kwargs)
        dims=Solver.find_size(piece_dataset, img)
        solution_dataset=Solver.dataset_from_img_split(img, dims)
        self.solved=self.arrange(solution_dataset, dims)
        return self.solved

    @staticmethod
    def parse_img(filename, type="png"):

        image_string = tf.read_file(filename)
        if type=="png":
            image_decoded = tf.image.decode_png(image_string)
        elif type=="jpg":
            image_decoded = tf.image.decode_jpeg(image_string)
        else: raise Exception(type)
        return image_decoded


    @staticmethod
    def pool(image_pipeline, size):
        assert image_pipeline.output_shapes.ndims==3 and image_pipeline.output_shapes.dims[3]==tf.Dimension(3)
        return image_dataset.batch(1).max_pool([1,size,size,1],[1,size,size,1],"VALID").map(lambdax: tf.squeeze(x, axis=0))


    @staticmethod
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


    @staticmethod
    def find_size(pieces, solution_img, **kwargs):
        #implement algorithm based on number of pieces, assuming square pieces, and solved image find_size
        return (3,3)


    @staticmethod
    def locator(locator_func, **kwarg):
        #return piece locator function
        def locator_wrapper(piece_dataset, solution_dataset, **kwarg):
            #do something and return ordered list with pieces+location
            return locator_func(piece_dataset, solution_dataset, **kwargs)
        return locator_wrapper


    @staticmethod
    def finder(finder_func):
        def finder_wrapper(img_path, **kwargs):
            piece_dataset=finder_func(img_path, **kwargs) #separator_func returns a list of images with IDS
            #piece_dataset=tf.data.dataset.zip((tf.data.Dataset.from_tensor_slices(tf.range(dims[0]*dims[1])), piece_dataset))
            return piece_dataset
        return separator_wrapper


    @staticmethod
    def arranger(arranger_func, **kwargs):
        def arranger_wrapper(piece_dataset, dims, **kwargs):
        #append pictures together and show them. May involve rotation and such.
            img=arranger_func(pieces, dims, **kwargs)
            return img
        return arranger_wrapper


    def __del__(self):
        pass



class ImplementThis(Exception):
    pass
