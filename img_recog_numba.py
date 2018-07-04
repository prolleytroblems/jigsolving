from numba import cuda,jit
import numpy as np
import cv2


class Solution(object):
    def __init__(self, path, dims):
        self.pieces=img_split(path, dims)
        self.dpieces=cuda.to_device(np.ascontiguousarray(self.pieces))
        self.locations=np.array([(i,j) for i in range(dims[1]) for j in range(dims[0])])
        self.availability=[True]*dims[0]*dims[1]


@cuda.jit(device=True)
def compare_pixel(pixela, pixelb):
    c=0
    for i in range(3):
        c+=(pixela[i]-pixelb[i])**2
    return c**0.5


@cuda.jit
def gcompare(imga, imgb, C):
    y, x = cuda.grid(2)
    if y>=imga.shape[0] or x>=imga.shape[1]:
        return

    C[y,x]=compare_pixel(imga[y,x], imgb[y,x])


def compare(dimga, dimgb, **params):
    tpb = 16
    bpgy = (dimga.shape[0]-1)//tpb+1
    bpgx = (dimga.shape[1]-1)//tpb+1

    C=np.array(np.zeros(dimga.shape[0:2]), dtype=np.float32)

    gcompare[(bpgy, bpgx), (tpb, tpb)](np.ascontiguousarray(dimga), np.ascontiguousarray(dimgb), C)

    return 1-np.sum(C)/(dimga.shape[0]*dimga.shape[1]*442)


def locate_one_piece(dpiece, solution, **params):
    """Will only receive preprocessed pieces!"""
    max_resemblance=(0, None)

    for i in range(solution.dpieces.shape[0]):
        if solution.availability[i]==True:
            resemblance=compare(dpiece, solution.dpieces[i], **params)
            if "debug_mode" in params:
                if params["debug_mode"]==True: print(resemblance)
            if resemblance>max_resemblance[0]:
                max_resemblance=(resemblance, i)
    solution.availability[max_resemblance[1]]=False
    return solution.locations[max_resemblance[1]]


def preprocess_pieces(pieces, solution_pieces, **params):
    if not(isinstance(pieces, np.ndarray) and isinstance(solution_pieces, np.ndarray)):raise TypeError("Wrong object type!")
    if not(len(pieces.shape)==4 and len(solution_pieces.shape)==4): raise Exception("Incorrect array shape!")
    if pieces.shape[0] != solution_pieces.shape[0]: raise Exception("Number of pieces don't match!")

    if pieces.shape != solution_pieces.shape:
        print("Piece shape mismatch!")
        pieces=resize(pieces, solution_pieces[0].shape[1:])
    return pieces


def resize(pieces, size):
    raise Exception("WRONG SIZE")


def locate_pieces(pieces, solution, **params):
    print(type(pieces), type(solution.pieces))
    pieces = preprocess_pieces(pieces, solution.pieces, **params)
    dpieces=cuda.to_device(np.ascontiguousarray(pieces))
    solved_locations=[]
    for i in range(len(solution.locations)):
        location=locate_one_piece(dpieces[i], solution, **params)
        solved_locations.append(location)
    return (pieces, np.array(solved_locations))


def brg_to_rgb(image):
    b,g,r=np.split(image, 3, axis=2)
    return np.concatenate((r,g,b), axis=2)


def img_split(image_path, dims, **params):
    assert type(dims)==tuple
    image=brg_to_rgb(cv2.imread(image_path, 1))

    pieces=[]
    height=image.shape[0]/dims[0]
    width=image.shape[1]/dims[1]
    for y_split in range(dims[0]):
        for x_split in range(dims[1]):
            x_start=int(x_split*width)
            x_end=x_start+int(width)
            y_start=int(y_split*height)
            y_end=y_start+int(height)

            pieces.append(np.array(image[y_start: y_end, x_start: x_end]))
    return np.array(pieces)


def sort_pieces(located_pieces, dims):
    sorted_pieces=[[0 for column in range(dims[1])] for row in range(dims[0])]
    for image, location in zip(located_pieces[0], located_pieces[1]):
        sorted_pieces[location[0]][location[1]]=image

    return [image for row in sorted_pieces for image in row]


def pool(image, pooling):
    pass


def main():
    pass


if __name__ == '__main__':
    main()
