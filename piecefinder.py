import cv2
import numpy as np
from datetime import datetime
from IoU import IoA
from utils import reflect, get_subarray, param_check, mili_seconds, find_dims

DEFAULTS = {"debug_mode":True, "ref_shape":None}

class PieceFinder(object):
    def __init__(self, **kwargs):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(cv2.getNumberOfCPUs())
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.filter=BBoxFilter(**kwargs)

    def find_boxes(self, array, base_k=150 ,inc_k=150, sigma=0.8, **kwargs):
        params=param_check(kwargs, DEFAULTS)
        if kwargs["debug_mode"]:
            print("-----Piece detector-----")

        start=datetime.now()
        self.ss.setBaseImage(array)
        self.ss.switchToSelectiveSearchFast(base_k ,inc_k , sigma)
        boxes=self.ss.process()
        if kwargs["debug_mode"]:
            print("Process: base_k={}, inc_k={}, sigma={}, {}".format(base_k , inc_k, sigma, mili_seconds(datetime.now()-start)))
            print("Received ", len(boxes), " proposals.")

        out = self.filter(array, boxes, **kwargs)

        return out

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

class BBoxFilter(object):

    def __init__(self, edgewidth=1, **kwargs):
        self.width=edgewidth
        self.configure(**kwargs)

    def configure(self, expansion=2, borderwidth=4, border_to_grad=0.4, threshold=(0.4,0.8),
                        max_loss=None, max_tries=10, **kwargs):
        self.expansion=expansion
        self.borderwidth=borderwidth
        self.weights=(border_to_grad, 1-border_to_grad)
        self.max_loss=max_loss
        self.max_tries=max_tries
        self.min_threshold=threshold[0]
        self.max_threshold=threshold[1]

    def __call__(self, array, boxes, ref_shape=None, **kwargs):
        start=datetime.now()

        subarrays=list(map(lambda box: get_subarray(array, box, self.expansion), boxes))
        scores=np.array(list(map(lambda subarray: self.score_array(subarray,
                                            thiccness=self.borderwidth), subarrays)))[...,None]
        del(subarrays)

        if not(self.max_loss is None):
            threshold = self.min_threshold
            step = abs(threshold-self.max_threshold)/self.max_tries
            found_values = np.zeros((self.max_tries, 4), dtype = object)
            for i in range(self.max_tries):
                out_boxes, out_scores = self.threshold_filter(boxes, scores, threshold)
                out_boxes, out_scores = self.IoA_filter(out_boxes, out_scores)

                if len(out_boxes)>0:
                    dims, loss = find_dims(self.average_shape(out_boxes), len(out_boxes), ref_shape)
                    found_values[i] = out_boxes, out_scores, dims, loss
                else:
                    found_values[i] = None, None, None, 1

                threshold+=step

            if loss > self.max_loss:
                out_boxes, out_scores, dims = found_values[np.argmin(found_values[:,3]), 0:3]
                #raise FindDimsFailure("Failed to find adequate boxes:" len(out_boxes))
            out=(out_boxes, out_scores, dims)
        else:
            threshold = self.threshold
            boxes, scores = self.threshold_filter(boxes, scores, threshold)
            out = self.IoA_filter(boxes, scores)


        if kwargs["debug_mode"]:
            if not(self.max_loss is None):
                print("Filter: threshold = {:5.3}, loss = {:5.3}, {}".format(threshold, loss, mili_seconds(datetime.now()-start)))
            else:
                print("Filter: threshold = {:5.3}, {}".format(threshold, mili_seconds(datetime.now()-start)))
        return out

    def average_shape(self, box_list):
        return np.average(np.array(box_list[:,(3,2)]), axis=0)

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

    def contrast_score(self, array, thiccness=2, exp_scaling=1.5, half_mark=20):
        MAKE BRIGHTNESS INVARIANT
        assert thiccness>1
        assert thiccness>self.expansion
        dirs=["N", "E", "S", "W"]
        score=0
        for dir in dirs:
            border = self.extract_border(array, thiccness, dir)
            base_value = np.sum(border[0], axis=0)/border[0].shape[0]


            subscore=0

            FIX THIS RANGE
            for i in range(self.expansion, thiccness):
                #channel-wise absolute difference of layer intensity averages
                lsubscore=np.absolute(base_value-border[i])
                #avg of absolute differences for each channel
                lsubscore=np.sum(lsubscore, axis=0)/lsubscore.shape[0]
                #average accross channels
                lsubscore=np.sum(lsubscore)/3
                #scaling the intensity difference value that gives score of 0.5
                lsubscore=lsubscore/half_mark
                IMPLEMENT DECREASING WEIGHTS
                #sigmoid normalization
                TRY DIFFERENT NORMALIZATION - XCORR?
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


class FindDimsFailure(Exception):
    pass
