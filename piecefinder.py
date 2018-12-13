import cv2
import numpy as np


class PieceFinder(object):
    def __init__(self, **kwargs):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(8)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.ss.switchToSelectiveSearchQuality()
        self.filter=BBoxFilter()


    def find_boxes(self, array):
        self.ss.setBaseImage(array)
        boxes=self.ss.process()
        boxes=self.filter(array, boxes)
        try to get dimensions

    def get_boxes(self, path, check_dims=False, iter=1 **kwargs):
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


    def configure(expansion=2, borderwidth=4, border_to_grad=0.5, **kwargs):
        self.expansion=expansion
        self.borderwidth
        self.weights=(border_to_grad, 1-border_to_grad)


    def __call__(self, array, boxes):
        subarrays=list(map(lambda box: self.get_subarray(array, box, 2), boxes))
        scores=list(map(lambda subarray: self.score_array(subarray), subarrays))



    def get_subarray(array, box, expansion=0):
        corners=[box[1]-expansion, box[1]+box[3]+expansion,
                 box[0]-expansion, box[0]+box[2]+expansion]
        if corners[0]<0:
            corners[0]=0
        if corners[2]<0:
            corners[2]=0

        return array[corners[0]:corners[1], corners[2]:corners[3]]


    def shrink_arrays(arrays, layers):
        return list(map(lambda array: array[layers:array.shape[0]-layers,
                                            layers:array.shape[1]-layers],
                        arrays))


    def score_array(self, array):
        b_score=self.border_score(array)
        c_score=self.contrast_score(array)
        score=b_score*self.weights[0]+c_score*self.weights[1]
        return score


    def extract_border(self, array, thiccness, dir):
        assert thiccness>0
        def extract_h(array, rows):
            return array[rows]

        def extract_v(array, collumns):
            return array[:,collumns].T

        return {"N": extract_h(array, list(range(thiccness))),
                "E": extract_v(array, list(range(-1, -1-thiccness, -1))),
                "S": extract_h(array, list(range(-1, -1-thiccness, -1))),
                "W": extract_v(array, list(range(thiccness)))}


    def extract_all_borders(self, array, thiccness):
        if thiccness==1:
            axis=0
            strip=np.ones((0, 3))
        else:
            axis=1
            strip=np.ones((thiccness, 0, 3))
        dirs=["N", "E", "S", "W"]
        for dir in dirs:
            next_strip=self.extract_border(array, thiccness, dir)
            np.concatenate((strip, next_strip), axis=axis)
        return strip


    def border_score(self, array, thiccness=1, color=np.array((255,255,255))):
        border = self.extract_all_borders(array, thiccness)
        avg=np.sum(border, axis=(0,1))/n
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
                lsubscore=np.sum(np.absolute(subscore))/3
                #scaling the intensity difference value that gives score of 0.5
                lsubscore=subscore/half_mark
                #sigmoid normalization
                lsubscore=subscore/(1+subscore)
                subscore+=lsubscore
            score+=subscore/4/thiccness-1
        return score
