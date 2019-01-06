import cv2
import numpy as np
from datetime import datetime
from IoU import IoA
from utils import reflect, get_subarray

class PieceFinder(object):
    def __init__(self, **kwargs):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(8)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.filter=BBoxFilter( **kwargs)

    def find_boxes(self, array):
        start=datetime.now()
        self.ss.setBaseImage(array)
        self.ss.switchToSelectiveSearchFast(base_k=200,inc_k=150, sigma=0.8)
        print("setup:", datetime.now()-start )
        start=datetime.now()
        boxes=self.ss.process()
        print("Received ", len(boxes), " proposals.")
        print("process:", datetime.now()-start )
        start=datetime.now()
        boxes, scores=self.filter(array, boxes)
        print("filter:", datetime.now()-start )
        return (boxes, scores)

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

    def configure(self, expansion=2, borderwidth=4, border_to_grad=0.4, threshold=0.6
    , **kwargs):
        self.expansion=expansion
        self.borderwidth=borderwidth
        self.weights=(border_to_grad, 1-border_to_grad)
        self.threshold=threshold

    def __call__(self, array, boxes):
        subarrays=list(map(lambda box: get_subarray(array, box, self.expansion), boxes))
        scores=np.array(list(map(lambda subarray: self.score_array(subarray,
                                            thiccness=self.borderwidth), subarrays)))[...,None]
        del(subarrays)

        boxes, scores = self.threshold_filter(boxes, scores, self.threshold)

        return self.IoA_filter(boxes, scores)

    def IoA_filter(self, boxes, scores, threshold=0.3):
        """Boxes have 4 parameters: (x0, y0, w, h)"""
        mask=np.ones((len(boxes)), dtype=np.bool)
        for i in range(len(boxes)):
            #print(i, mask)
            if mask[i]:
                for j in range(i+1, len(boxes)):
                    ioa=IoA(boxes[i], boxes[j])
                    if any([i>threshold for i in ioa]):
                        if scores[i]>scores[j]:
                            mask[j]=False
                        else:
                            mask[i]=False
                            break
        return (boxes[mask,...], scores[mask])

    def threshold_filter(self, boxes, scores, threshold):
        mask=np.ones((len(boxes)), dtype=np.bool)
        for i, score in enumerate(scores):
            if score<threshold:
                mask[i]=False
        return (boxes[mask,...], scores[mask])

    def size_filter(self, box_shape, min_size):
        if box_shape[0]<min_size:
            return False
        if box_shape[1]<min_size:
            return False
        return True

    def shrink_array(self, array, layers):
        return array[layers:array.shape[0]-layers, layers:array.shape[1]-layers]

    def score_array(self, array, thiccness):
        if not(self.size_filter(array.shape, thiccness*2)):
            return 0
        b_score=self.border_score(array, thiccness=1)
        c_score=self.contrast_score(array, thiccness=thiccness)

        score=b_score*self.weights[0]+c_score*self.weights[1]
        #print(b_score, c_score, score)
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
        return strip

    def border_score(self, array, thiccness=1, color=np.array((255,255,255)), half_mark=15):
        border = self.extract_all_borders(array, thiccness)
        avg=np.sum(border, axis=(0,1))/(border.shape[0]*border.shape[1])
        avg_error=np.sum(np.absolute(color-avg))/3

        score=1/(avg_error/half_mark+1)
        return score

    def contrast_score(self, array, thiccness=2, exp_scaling=3, half_mark=20):
        assert thiccness>1
        assert thiccness>self.expansion
        dirs=["N", "E", "S", "W"]
        score=0
        for dir in dirs:
            border = self.extract_border(array, thiccness, dir)
            base_value = np.sum(border[0], axis=0)/border[0].shape[0]


            subscore=0
            #IMPLEMENT DECREASIN WEIGHTS
            for i in range(self.expansion, thiccness):
                #channel-wise absolute difference of layer intensity averages
                lsubscore=np.absolute(base_value-border[i])
                #avg of absolute differences for each channel
                lsubscore=np.sum(lsubscore, axis=0)/lsubscore.shape[0]
                #average accross channels
                lsubscore=np.sum(lsubscore)/3
                #scaling the intensity difference value that gives score of 0.5
                lsubscore=lsubscore/half_mark
                #sigmoid normalization
                TRY DIFFERENT NORMALIZATION
                lsubscore=lsubscore/(1+lsubscore)
                subscore+=lsubscore
            """print(subscore)
            cv2.imshow("YOLO", cv2.resize(border, None, fx=1, fy=10))
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            score+=subscore/4/(thiccness-self.expansion)
        return score

    def sample_columns(self, array, samples):
        columns = np.random.choice(array.shape[1], samples, replace=False)
        return array[:, columns]
