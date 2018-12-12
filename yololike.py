from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import chainer
from IoU import *
import functools
from datetime import datetime
import cupy as cp

from chainer.backends.cuda import to_cpu

import cv2

def batchify(type="variable"):
    def batchify_decorator(func):
        @functools.wraps(func)
        def batch_func(self, batch, *args, **kwargs):
            out=[]
            for x in batch:
                out.append(func(self, x, *args, *kwargs))
            if type=="variable":
                return F.stack(tuple(out))
            elif type=="list":
                return out
            else:
                 raise Exception()
        return batch_func
    return batchify_decorator


class YoloLike(chainer.Chain):
    def __init__(self, S=7, B=2, input_size=(1024,768,3)):
        super().__init__()
        self.S=S
        self.B=B
        with self.init_scope():
            self.CNN=YoloBlock(B=self.B)
        self.gen_references(input_size)
        self.input_size=input_size
        self.CNN_input_size=(input_size[2], input_size[1], input_size[0])
        self.pred_cache=None


    def split_image(self, x):
        splits=np.zeros((self.S, self.S), dtype=object)
        split_rows=np.array_split(x, self.S, 2)
        for y_index, split_row in enumerate(split_rows):
            for x_index, split in enumerate(np.array_split(split_row, self.S, 3)):
                splits[y_index, x_index]=split
        return splits


    def gen_references(self, input_size):
        self.references=np.zeros((self.S, self.S, 4), dtype=np.int32)
        longs=(0,0)
        for dim in (0,1):
            longs = input_size[dim] % self.S
            longs_len = input_size[dim] // self.S +1
            cumulative=0
            for index in range(self.S):
                self._ref_slices(cumulative, dim, index, 0+dim)
                if index<longs:
                    self._ref_slices(longs_len, dim, index, 2+dim)
                    cumulative +=longs_len
                else:
                    self._ref_slices(longs_len-1, dim, index, 2+dim)
                    cumulative +=longs_len-1


    def _ref_slices(self, value, dim, slice_dim, coord):
        if dim==0:
            self.references[slice_dim, :, coord] = value
        elif dim==1:
            self.references[:, slice_dim, coord] = value
        else:
            raise Exception()


    @batchify(type="list")
    def IoA_filter(self, boxes, threshold=0.3):
        """boxes have 5 parameters"""
        predictions=boxes.data
        mask=np.ones((len(predictions)), dtype=np.bool)
        for i in range(len(predictions)):
            #print(predictions)
            if mask[i]:
                for j in range(i+1, len(predictions)):
                    ioa=IoA(predictions[i,:4], predictions[j,:4])
                    if any([i>threshold for i in ioa]):
                        if predictions[i,4]>predictions[j,4]:
                            mask[j]=False
                        else:
                            mask[i]=False
                            break
        result=boxes[mask,...]
        return result


    @batchify()
    def rescale_boxes(self, predictions, grid_coord):
        ref=self.references[grid_coord]
        mult_array=[[ref[2], ref[3], ref[2], ref[3], 1]]*self.B
        mult_array=np.asarray(mult_array)
        add_array=np.asarray([[ref[0], ref[1], 0, 0, 0]]*self.B)
        boxes=predictions*mult_array+add_array
        return boxes


    @batchify(type="list")
    def threshold_filter(self, predictions, threshold):
        mask=np.ones((len(predictions)), dtype=np.bool)
        for i, prediction in enumerate(predictions):
            if prediction.data[4]<threshold:
                mask[i]=False
        return predictions[mask,...]


    def split_1d_variable(self, variable, splits):
        n=variable.shape[1]//splits
        assert variable.shape[1]%splits==0
        return F.reshape(variable,(variable.shape[0], self.B, 5))


    def predict(self, x):
        assert x.shape[1:]==self.CNN_input_size
        splits=self.split_image(x)
        predictions=chainer.Variable(np.zeros((x.shape[0],0,5), dtype=np.float32))
        for y_index in range(self.S):
            for x_index in range(self.S):
                grid_coord=(y_index, x_index)
                prediction=self.CNN(splits[grid_coord])
                #print(prediction, "cnn")
                prediction=self.split_1d_variable(prediction, self.B)
                #print(prediction, "split")
                prediction=self.rescale_boxes(prediction, grid_coord)
                #print(prediction, "rescale")
                #reshaped=np.reshape(np.asarray(prediction), (self.B*len(prediction), 5))
                predictions=F.concat((predictions, prediction), axis=1)

        """image1=np.asarray(x[0]*255, dtype=np.int16)
        image1=np.asarray(image1, dtype=np.uint8)
        image1=np.stack((image1[0], image1[1], image1[2]), axis=-1)
        image2=np.array(image1)
        image3=np.array(image1)

        for box in predictions[0]:
            data=box.data
            cv2.rectangle(image1, (data[0],data[1]), (data[0]+data[2], data[1]+data[3]), color=(255,255,0), thickness=4)
        cv2.imwrite("pre-filter.jpg", image1)"""

        predictions=self.threshold_filter(predictions, 0.6)
        """for box in predictions[0]:
            data=box.data
            cv2.rectangle(image3, (data[0],data[1]), (data[0]+data[2], data[1]+data[3]), color=(255,255,0), thickness=4)
        cv2.imwrite("threshold_filter.jpg", image3)"""

        predictions=self.IoA_filter(predictions)
        """for box in predictions[0]:
            data=box.data
            cv2.rectangle(image2, (data[0],data[1]), (data[0]+data[2], data[1]+data[3]), color=(255,255,0), thickness=4)
        cv2.imwrite("IoA_filter.jpg", image2)"""
        return predictions

    def forward(self, x):
        boxes=self.predict(x)
        self.pred_cache=boxes
        return boxes

    def loss(self, x, truths, debug_mode=True):
    #REWRITE THIS IN A MUCH BETTER WAY (percentage of uncovered pieces??)
        start=datetime.now()
        predictions=self(x)
        #print("forward", datetime.now()-start)

        start=datetime.now()
        truths=self.unpad(truths)
        pt_pair=[(predictions[i][:,0:4], truths[i]) for i in range(x.shape[0])]
        iou=self.IoU(pt_pair)
        #print("IoU loss", datetime.now()-start)

        loss=1-sum(iou)
        return loss

    @batchify(type="list")
    def IoU(self, list_pair):
        listA, listB = list_pair
        return list_IoU(listA, listB, list_type=chainer.Variable)

    def unpad(self, boxes):
        mask=np.ones((len(boxes)), dtype=np.bool)
        for i, box in enumerate(boxes):
            if box[0]<0
                mask[i]=False
        return boxes[mask]

    def accuracy(self, truths, x=None):
        if x:
            pred=self(x)
        else:
            pred=self.pred_cache
        return sum(self.IoU((self.pred_cache, truths)))

class YoloBlock(Chain):
    def __init__(self, B):
        super().__init__()
        self.B=B
        w=chainer.initializers.HeNormal()
        with self.init_scope():
            self.c1=L.Convolution2D(in_channels=3, out_channels=24, ksize=3, stride=1, pad=0, initialW=w)
            self.c2=L.Convolution2D(in_channels=None, out_channels=48, ksize=3, stride=1, pad=0, initialW=w)
            self.c3=L.Convolution2D(in_channels=None, out_channels=96, ksize=3, stride=1, pad=0, initialW=w)
            self.c4=L.Convolution2D(in_channels=None, out_channels=96, ksize=3, stride=1, pad=0, initialW=w)
            self.c5=L.Convolution2D(in_channels=None, out_channels=192, ksize=3, stride=1, pad=0, initialW=w)
            self.c6=L.Convolution2D(in_channels=None, out_channels=192, ksize=3, stride=1, pad=0, initialW=w)
            self.f7 = L.Linear(None, 64)
            self.f8 = L.Linear(64, self.B*5)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.c3(h))
        h = F.relu(self.c4(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.c5(h))
        h = F.relu(self.c6(h))
        h = F.dropout(F.relu(self.f7(h)))
        h = F.sigmoid(self.f8(h))
        return h
