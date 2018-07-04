from numba import cuda,jit
import numpy as np
import cv2


class Solution(object):
    def __init__(self, path, dims):
        self.pieces=img_split(path, dims)
        self.dpieces=cuda.to_device(np.ascontiguousarray(self.pieces))
        self.locations=np.array([(i,j) for i in range(dims[0]) for j in range(dims[1])])
        self.availability=[True]*dims[0]*dims[1]
        self.shape=dims


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


def preprocess_pieces(pieces, solution, pooling=None, **params):
    if not(isinstance(pieces, np.ndarray) and isinstance(solution, Solution)):raise TypeError("Wrong object type!")
    if not(len(pieces.shape)==4 and len(solution.pieces.shape)==4): raise Exception("Incorrect array shape!")
    if pieces.shape[0] != solution.pieces.shape[0]: raise Exception("Number of pieces don't match!")

    if pieces.shape != solution.pieces.shape:
        print(pieces.shape, solution.pieces.shape)
        print("Piece shape mismatch!")
        pieces=resize_batch(pieces, (solution.pieces[0].shape[0:2][::-1]))

    if pooling != None:
        pieces=pool(pieces, (5,5), (5,5))
        solution=pool(solution, (5,5), (5,5))

    return (pieces, solution)


def resize_batch(pieces, size):
    resized=[]
    for piece in pieces:
        resized.append(cv2.resize(piece, size))
    return np.array(resized)


def locate_pieces(pieces, solution, pooling=None, **params):
    pieces, solution = preprocess_pieces(pieces, solution, pooling, **params)
    dpieces = cuda.to_device(np.ascontiguousarray(pieces))
    solved_locations=[]
    for i in range(len(solution.locations)):
        location=locate_one_piece(dpieces[i], solution, **params)
        solved_locations.append(location)
    return (pieces, np.array(solved_locations))


def full_solve(pieces, solution, pooling=None, **params):

    return reassemble(sort_pieces(locate_pieces(pieces, solution, pooling, **params), solution.shape), solution.shape)


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


def reassemble(pieces, dims):
    """Reassembles ordered piece images into a full image"""
    pieces=np.array(pieces)
    if not(len(pieces.shape)==4): raise TypeError("pieces must be a 4-dimensional ndarray-like object")
    if not(isinstance(dims, tuple) and len(dims)==2): raise TypeError("dims not legible as tuple")
    image=np.concatenate([np.concatenate(pieces[i*dims[1]:(i+1)*dims[1]], axis=1) for i in range(dims[0])], axis=0)
    return image


@cuda.jit
def max_pool_unit(image, pooling, stride, pooled_image):
    y,x=cuda.grid(2)

    if y>pooled_image.shape[0] or x>pooled_image.shape[1]:
        return
    window=image[y*stride[1]:y*stride[1]+pooling[1], x*stride[0]:x*stride[0]+pooling[0], :]

    pooled_image[y,x,0]=0
    pooled_image[y,x,1]=0
    pooled_image[y,x,2]=0

    for i in range(pooling[0]):
        for j in range(pooling[1]):
            if window[i,j,0]>pooled_image[y,x,0]:
                pooled_image[y,x,0]=window[i,j,0]
            if window[i,j,1]>pooled_image[y,x,1]:
                pooled_image[y,x,1]=window[i,j,1]
            if window[i,j,2]>pooled_image[y,x,2]:
                pooled_image[y,x,2]=window[i,j,2]


def pool_image(image, pooling, stride):
    assert pooling[0]>=stride[0] and pooling[1]>=stride[1]
    def add_padding(image, axis, side="end"):
        if (image.shape[axis]-pooling[axis]+stride[axis])%stride[axis]==0:
            return image
        else:
            #add a blank row/column
            coeff=(0,0)
            coeff[axis]=1
            if side=="end":
                np.concatenate(image, np.zeros(((image.shape[0]-1)*coeff[0]+1, (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8), axis=axis)
                image=add_padding(image, axis, side="start")
            elif side=="start":
                np.concatenate(np.zeros(((image.shape[0]-1)*coeff[0]+1, image, (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8), axis=axis)
                image=add_padding(image, axis, side="end")
            return image

    for axis in range(2):
        image=add_padding(image, axis, "end")

    final_dims=((image.shape[0]-pooling[0]+stride[0])//stride[0], (image.shape[1]-pooling[1]+stride[1])//stride[1])
    tpb=16
    bpgy=(final_dims[0]-1)//tbp+1
    bpgy=(final_dims[1]-1)//tbp+1

    dimage=cuda.to_device(np.ascontiguousarray(image))
    dpooled=cuda.to_device(np.ndarray(np.zeros(final_dims), dtype=np.uint8))
    max_pool_unit[(), ()](dimage, pooling, stride, )



def main():
    pass


if __name__ == '__main__':
    main()
