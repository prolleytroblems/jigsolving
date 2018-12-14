import cv2
import numpy as np
from datetime import datetime
from IoU import IoA
from utils import reflect

class PieceFinder(object):
    def __init__(self, **kwargs):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(8)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.filter=BBoxFilter()


    def find_boxes(self, array):
        start=datetime.now()
        self.ss.setBaseImage(array)
        self.ss.switchToSelectiveSearchFast()
        print("setup:", datetime.now()-start )
        start=datetime.now()
        boxes=self.ss.process()
        print("process:", datetime.now()-start )
        start=datetime.now()
        boxes=self.filter(array, boxes)
        print("filter:", datetime.now()-start )
        start=datetime.now()
        return boxes

    def get_boxes(self, path, check_dims=False, iter=1, **kwargs):
        if iter>n:
            raise Exception("Could not find good boxes with given restrictions.")

        array = cv2.imread(path.tostr)

        box_list = self.find_boxes(array)

        if check_dims:
            full_shape=kwargs["full_shape"]
            similarity = self.check_dims(box_list, full_shape)
            if similarity > something:
                return box_list
            else:
                return self.get_boxes(path, check_dims=True, full_shape=full_shape, iter=iter+1)
        else:
            return box_list

    def check_dims(self, box_list, full_shape):
        pass


class BBoxFilter(object):

    def __init__(self, edgewidth=1, **kwargs):
        self.width=edgewidth
        self.configure(**kwargs)

    def configure(self, expansion=2, borderwidth=4, border_to_grad=0.5, **kwargs):
        self.expansion=expansion
        self.borderwidth=borderwidth
        self.weights=(border_to_grad, 1-border_to_grad)

    def __call__(self, array, boxes):
        subarrays=list(map(lambda box: self.get_subarray(array, box, self.expansion), boxes))
        scores=np.array(list(map(lambda subarray: self.score_array(subarray,
                                            thiccness=self.borderwidth), subarrays)))[...,None]
        #print(scores)
        del(subarrays)

        return self.IoA_filter(boxes, scores)

    def IoA_filter(self, boxes, scores, threshold=0.3):
        """Boxes have 5 parameters: (x0, y0, w, h, score)"""
        mask=np.ones((len(boxes)), dtype=np.bool)
        for i in range(len(boxes)):
            if mask[i]:
                for j in range(i+1, len(boxes)):
                    ioa=IoA(boxes[i], boxes[j])
                    if any([i>threshold for i in ioa]):
                        if scores[i]>scores[j]:
                            mask[j]=False
                        else:
                            mask[i]=False
                            break
        result=boxes[mask,...]
        return result

    def get_subarray(self, array, box, expansion=0):
        corners=[box[1]-expansion, box[1]+box[3]+expansion,
                 box[0]-expansion, box[0]+box[2]+expansion]
        if corners[0]<0:
            corners[0]=0
        if corners[2]<0:
            corners[2]=0

        return array[corners[0]:corners[1], corners[2]:corners[3]]

    def shrink_array(self, array, layers):
        return array[layers:array.shape[0]-layers, layers:array.shape[1]-layers]

    def score_array(self, array, thiccness):
        b_score=self.border_score(array, thiccness=1)
        c_score=self.contrast_score(array, thiccness=thiccness)
        print(b_score, c_score)
        score=b_score*self.weights[0]+c_score*self.weights[1]
        return score

    def extract_border(self, array, thiccness, dir):
        assert thiccness>0
        def extract_h(array, rows):
            return array[rows]

        def extract_v(array, collumns):
            out=array[:,collumns]
            return reflect(out)

        return {"N": lambda x: extract_h(x, list(range(thiccness))),
                "E": lambda x: extract_v(x, list(range(-1, -1-thiccness, -1))),
                "S": lambda x: extract_h(x, list(range(-1, -1-thiccness, -1))),
                "W": lambda x: extract_v(x, list(range(thiccness)))}[dir](array)

    def extract_all_borders(self, array, thiccness):
        strip=np.ones((thiccness, 0, 3), dtype=np.uint8)
        dirs=["N", "E", "S", "W"]
        for dir in dirs:
            next_strip=self.extract_border(array, thiccness, dir)
            strip=np.concatenate((strip, next_strip), axis=1)
        #print(strip)
        return strip

    def border_score(self, array, thiccness=1, color=np.array((255,255,255))):
        border = self.extract_all_borders(array, thiccness)
        avg=np.sum(border, axis=(0,1))/(array.shape[0]*array.shape[1])
        square_error=np.sum((avg-color)**2)
        score=1/(square_error+1)
        return score

    def contrast_score(self, array, thiccness=2, exp_scaling=2, half_mark=10):
        assert thiccness>1
        dirs=["N", "E", "S", "W"]
        score=0
        for dir in dirs:
            border = self.extract_border(array, thiccness, dir)
            base_value = np.sum(border[0], axis=0)
            subscore=0
            for i in range(thiccness-1):
                #channel-wise difference of layer intensity averages
                lsubscore=np.sum(border[i+1], axis=0)-base_value
                #avg of absolute differences across channels
                lsubscore=np.sum(np.absolute(lsubscore))/3
                #scaling the intensity difference value that gives score of 0.5
                lsubscore=lsubscore/half_mark
                #sigmoid normalization
                lsubscore=lsubscore/(1+lsubscore)
                subscore+=lsubscore

            "subscores are 0"
            score+=subscore/4/(thiccness-1)
            print(score)
        #print(score)
        return score
