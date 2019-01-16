import cv2
from utils import *
from dataset import *
from pathlib import Path
from img_recog_proto import *
from img_recog_numba import *
from numba import cuda
import numpy as np


paths = ["./thesis/2018-06-08 001.JPG", "./thesis/2018-09-05 058.JPG", "./thesis/2018-09-05 069.JPG"]
dims = (2,2)


paths_a = ["./thesis/sky/2018-04-08 034-9.jpg", "./thesis/sky/2018-04-08 034-4.jpg", "./thesis/sky/2018-04-08 034-9.jpg", "./thesis/sky/2018-04-08 034-4.jpg"]
paths_b = ["./thesis/sky/2018-04-08 034-9-gaussian.jpg", "./thesis/sky/2018-04-08 034-9-gaussian.jpg", "./thesis/sky/2018-04-08 034-4-gaussian.jpg", "./thesis/sky/2018-04-08 034-4-gaussian.jpg"]

out = open("./thesis/sky/comparisons-xcorr.txt", "w")
for path_a, path_b in zip(paths_a, paths_b):
    imga=cv2.imread(path_a)
    imgb=cv2.imread(path_b)
    dimga=cuda.to_device(np.ascontiguousarray(imga))
    dimgb=cuda.to_device(np.ascontiguousarray(imgb))

    comparison = compare_xcorr(imga, imgb, dimga, dimgb)
    print(comparison)
    out.write(path_a+", "+path_b+", "+str(comparison)+"\n")

out.close()


"""paths = ["./thesis/sky/2018-04-08 034-9.jpg", "./thesis/sky/2018-04-08 034-4.jpg"]
for i, path in enumerate(paths):
    out=path[:-4]+"-gaussian.jpg"
    img=cv2.imread(path)
    cv2.imwrite(out, distort(img, 10, "g"))"""

"""PATH="./thesis/2018-04-08 034.JPG"
splits=img_split(PATH, (5,5), invert = False)
for i, split in enumerate(splits):
    cv2.imwrite("./thesis/sky/2018-04-08 034-"+str(i)+".jpg", split)"""

"""pooled = pool([cv2.imread("./thesis/2018-06-08 001-0.jpg")], (7,7), (7,7))[0]
cv2.imwrite("./thesis/2018-06-08 001-0-pooled.jpg", pooled)"""


"""img = np.ones((200, 200, 3), dtype=np.uint8)*255
cv2.circle(img, (100,100), 60, thickness = -1, color=(0, 0, 0))
out="./thesis/circle.jpg"
cv2.imwrite(out, img)

img = np.ones((200, 200, 3), dtype=np.uint8)*255
cv2.circle(img, (100,100), 60, thickness = -1, color=(0, 0, 0))
out="./thesis/circle-gaussian.jpg"
cv2.imwrite(out, distort(img, 15, "g"))

img = np.ones((200, 200, 3), dtype=np.uint8)*255
cv2.circle(img, (100,100), 60, thickness = -1, color=(0, 0, 0))
out="./thesis/circle-brightness.jpg"
cv2.imwrite(out, distort(img, 30, "b"))

img = np.ones((200, 200, 3), dtype=np.uint8)*255
cv2.circle(img, (100,100), 60, thickness = -1, color=(0, 0, 0))
out="./thesis/circle-motion.jpg"
cv2.imwrite(out, distort(img, 15, "m"))

img = np.ones((200, 200, 3), dtype=np.uint8)*255
cv2.circle(img, (100,100), 60, thickness = -1, color=(0, 0, 0))
out="./thesis/circle-blur.jpg"
cv2.imwrite(out, distort(img, 15, "bl"))"""

"""splitter = ImageSplitter(txtpath="./thesis/truthvalues.json")
for path in paths:
    splitter.gen(Path(path), Path(path).parent/"scrambled", dims=(2,2), min=-1)"""
