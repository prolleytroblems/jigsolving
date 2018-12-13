import cv2
import numpy as np


class PieceFinder(object):
    def __init__(self, **kwargs):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(8)
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.ss.switchToSelectiveSearchQuality()
        self.filter=BBoxFilter()


    def find_boxes(self, image):
        self.ss.setBaseImage(image)
        boxes=self.ss.process()
        boxes=self.filter(image, boxes)
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

    def __init__(self, edgewidth=1):
        self.width=edgewidth

    def configure(expansion=2, borderwidth=4, border_to_grad=0.5):
        self.expansion=expansion
        self.borderwidth
        self.weights=(border_to_grad, 1-border_to_grad)

    def __call__(self, image, boxes):
        pass


    def check(self, image):
        pass

    def score_image(self, image):

        score=border_score*self.weights[0]+grad_score*self.weights[1]
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

    def border_score(self, image, thiccness=1, color=np.array((255,255,255))):
        border = self.extract_all_borders(image, thiccness)
        avg=np.sum(border, axis=(0,1))/n
        square_error=np.sum((avg-color)**2)
        score=1/(square_error+1)

    def contrast_score(self):
        pass
