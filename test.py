from piecefinder import *
import cv2
from utils import *

finder = PieceFinder()
print("1")
array=cv2.imread("./images/totalbiscuit.jpg")
array=cv2.resize(array, None, fx=0.8, fy=0.8)
boxes=finder.find_boxes(array)


for box in boxes:
    x, y, w, h = box
    cv2.rectangle(array, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow("YOLO", array)
cv2.waitKey(0)
cv2.destroyAllWindows()
