from piecefinder import *
import cv2
from utils import *
import numpy as np

finder = PieceFinder()
array=cv2.imread("./images/totalbiscuit.jpg")
array=cv2.imread(".\images\DSC_0501.JPG")
array=cv2.resize(array, None, fx=0.5, fy=0.5)

filter=BBoxFilter()
boxes, scores=finder.find_boxes(array)
for box, score in zip(boxes, scores):
    x, y, w, h = box
    cv2.rectangle(array, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(array, str(score), (x, y+h), 1, 1, (0,0,0))

"""box=np.array((264,316,87,99))
x,y,w,h =box
box, score= filter(array, box[None])
subim=np.array(array[y:y+h,x:x+w])
cv2.rectangle(array, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(array, str(score[0]), (x, y+h), 1, 1, (0,0,0))"""

cv2.imshow("YOLO", array)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""cv2.imshow("YOLO",subim )
cv2.waitKey(0)
cv2.destroyAllWindows()"""
