from dataset import *
from utils import *
from img_recog_proto import *
from filters import *
import os
from pathlib import Path

def make_base(path):
    genner = ImageSplitter(txtpath = str(Path(path) / "truthvalues.json"))
    genner.gen_all(path, dims = (4,6))
    genner.close()


def make_distort(scrambledpath, outpath, dfunc, dev):
    scrambledpath= Path(scrambledpath)
    genner = ImageSplitter(txtpath = "")
    images = genner.find_images(path=scrambledpath, extension=".jpg")
    outpath=Path(outpath)
    if not outpath.exists():
        os.mkdir(outpath)
    for image in images:
        array=cv2.imread(str(image))
        array=dfunc(array, dev)
        cv2.imwrite(str(outpath/str(image.name)), array)

def make_all_dist(path):
    path = Path(path)
    d_funcs={"blur" :lambda x, dev: gaussian_blur(x, stddev=dev),
             "motion" :lambda x, k: motion_blur(x, kernel_size=k),
             "brightness" :lambda x, inc: distort(x, inc, "b"),
             "gaussian" :lambda x, dev: distort(x, dev, "g")}
    for dev in [25,50,75,100]:
        make_distort(path, path.parent/str("brightness-"+str(dev)+"/"), d_funcs["brightness"], dev)
    for dev in [5,9,13,17]:
        make_distort(path, path.parent/str("motion-"+str(dev)+"/"), d_funcs["motion"], dev)
    for dev in [1,2,3,4]:
        make_distort(path, path.parent/str("blur-"+str(dev)+"/"), d_funcs["blur"], dev)
    for dev in [5,10,15,20]:
        make_distort(path, path.parent/str("gaussian-"+str(dev)+"/"), d_funcs["gaussian"], dev)

if __name__ == "__main__":
    make_base("./tests/")
    make_all_dist("./tests/samples")
