from numba import cuda,jit
from datetime import datetime
import numpy as np
import cv2


class Solution(object):
    def __init__(self, path_or_pieces, dims):
        if isinstance(path_or_pieces, np.ndarray) and len(path_or_pieces.shape)==4:
            self.pieces=path_or_pieces
        elif isinstance(path_or_pieces, str):
            self.pieces = img_split(path_or_pieces, dims)
        else:
            try:
                 self.pieces=np.array(pieces)
                 if self.pieces.shape!=4: raise Exception
            except Exception as e:
                raise TypeError("path_or_pieces must be a path, or 4D ndarray-like of piece pbjects")
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
    if not("debug_mode" in params):
        params["debug_mode"]=False
    if not("threshold" in params):
        params["threshold"]=None

    if params["debug_mode"]==True:


    max_resemblance=[0, None, 0, 1]
    #maximum resemblance, location index, second max resemblance(for debugging), min resemblance (for debugging)

    for i in range(solution.dpieces.shape[0]):
        if solution.availability[i]==True:
            resemblance=compare(dpiece, solution.dpieces[i], **params)

            if resemblance>max_resemblance[0]:
                if params["debug_mode"]==True:
                    max_resemblance[2]=max_resemblance[0]
                    if resemblance<max_resemblance[3]:
                        max_resemblance[3]=resemblance
                max_resemblance[0]=resemblance
                max_resemblance[1]=i
                if params["threshold"]!=None:
                    if resemblance>params["threshold"]:
                        if params["debug_mode"]==True:
                            print("Threshold crossed")
                        break
            elif params["debug_mode"]==True:
                if resemblance>max_resemblance[2]:
                    max_resemblance[2]=resemblance
                if resemblance<max_resemblance[3]:
                    max_resemblance[3]=resemblance

    solution.availability[max_resemblance[1]]=False

    if params["debug_mode"]==True:
        print("Piece index: "+str(params["index"])+
                ", Max res.: "+ str(max_resemblance[0])+
                ", 2nd max res.: "+ str(max_resemblance[2])+
                ", Min res.: "+ str(max_resemblance[3])+
                ", ")

    return solution.locations[max_resemblance[1]]


def preprocess_pieces(pieces, solution, pooling=None, **params):
    if not("debug_mode" in params):
        params["debug_mode"]=False
    if params["debug_mode"]==True:
        start=datetime.now()

    if not(isinstance(pieces, np.ndarray) and isinstance(solution, Solution)):raise TypeError("Wrong object type!")
    if not(len(pieces.shape)==4 and len(solution.pieces.shape)==4): raise Exception("Incorrect array shape!")
    if pieces.shape[0] != solution.pieces.shape[0]: raise Exception("Number of pieces don't match!")

    if pieces.shape != solution.pieces.shape:
        print(pieces.shape, solution.pieces.shape)
        print("Piece shape mismatch!")
        pieces=resize_batch(pieces, (solution.pieces[0].shape[0:2][::-1]), **params)

    if pooling != None and pooling != 1:
        pieces=pool(pieces, (pooling, pooling), (pooling, pooling))
        solution=pool(solution, (pooling, pooling), (pooling, pooling))

    if params["debug_mode"]==True:
        print("Preprocessing: "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return (pieces, solution)


def resize_batch(pieces, size, **params):
    resized=[]
    for piece in pieces:
        resized.append(cv2.resize(piece, size))
    return np.array(resized)


def locate_pieces(pieces, solution, pooling=None, **params):
    if not("debug_mode" in params):
        params["debug_mode"]=False

    p_pieces, p_solution = preprocess_pieces(pieces, solution, pooling, **params)
    dpieces = cuda.to_device(np.ascontiguousarray(p_pieces))

    solved_locations=[]
    for i in range(len(solution.locations)):
        if params["debug_mode"]==True:
            params["index"]=i

        SHOULD ITERATE OVER LOCATIONS, NOT PIECES
        location=locate_one_piece(dpieces[i], p_solution, **params)
        solved_locations.append(location)
    return (pieces, np.array(solved_locations))


def full_solve(pieces, solution, pooling=None, **params):
    if not("debug_mode" in params):
        params["debug_mode"]=False

    if params["debug_mode"]==True:
        start=datetime.now()

    solved=reassemble(sort_pieces(locate_pieces(pieces, solution, pooling, **params), solution.shape), solution.shape)

    if params["debug_mode"]==True:
        print("Solving: "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return solved


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
def max_pool_unit(image, pooling, stride, pooled):
    y, x=cuda.grid(2)

    if y>pooled.shape[0] or x>pooled.shape[1]:
        return
    window=image[y*stride[1]:y*stride[1]+pooling[1], x*stride[0]:x*stride[0]+pooling[0], :]

    pooled[y,x,0]=0
    pooled[y,x,1]=0
    pooled[y,x,2]=0

    for i in range(pooling[0]):
        for j in range(pooling[1]):
            if window[i,j,0]>pooled[y,x,0]:
                pooled[y,x,0]=window[i,j,0]
            if window[i,j,1]>pooled[y,x,1]:
                pooled[y,x,1]=window[i,j,1]
            if window[i,j,2]>pooled[y,x,2]:
                pooled[y,x,2]=window[i,j,2]


def pool(images_or_solution, pooling, stride, **params):
    if isinstance(images_or_solution, np.ndarray):
        pooled=[]
        for image in images_or_solution:
            pooled.append(pool_image(image, pooling, stride, **params))
        return np.array(pooled, dtype=np.uint8)

    elif isinstance(images_or_solution, Solution):
        dims=images_or_solution.shape
        images_or_solution = images_or_solution.pieces
        pooled=[]
        for image in images_or_solution:
            pooled.append(pool_image(image, pooling, stride, **params))
        return Solution(np.array(pooled, dtype=np.uint8), dims)

    else: raise TypeError("images_or_solution must be an ndarray or Solution instance")




def pool_image(image, pooling, stride):
    """Apply pooling to a single image. \n
        image    """
    if not(pooling[0]>=stride[0] and pooling[1]>=stride[1]): raise TypeError("Pooling and stride inputs should be ndarray-like")
    if not(len(image.shape)==3): raise TypeError("Image must be a 3D ndarray.")

    pooling=np.array(pooling, dtype=np.uint8)
    stride=np.array(stride, dtype=np.uint8)

    def add_padding(image, axis, side="end"):
        assert axis==0 or axis==1
        if (image.shape[axis]-pooling[axis]+stride[axis])%stride[axis]==0:
            return image
        else:
            #add a blank row/column
            coeff=[0,0]
            coeff[-axis+1]=1
            if side=="end":
                image=np.concatenate((image, np.zeros(((image.shape[0]-1)*coeff[0]+1,
                        (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8)), axis=axis)
                image=add_padding(image, axis, side="start")
            elif side=="start":
                image=np.concatenate((np.zeros(((image.shape[0]-1)*coeff[0]+1,
                        (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8), image), axis=axis)
                image=add_padding(image, axis, side="end")
            return image

    for axis in range(2):
        image=add_padding(image, axis, "end")

    final_dims=((image.shape[0]-pooling[0]+stride[0])//stride[0], (image.shape[1]-pooling[1]+stride[1])//stride[1], 3)
    tpb=16
    bpgy=(final_dims[0]-1)//tpb+1
    bpgx=(final_dims[1]-1)//tpb+1

    dimage=cuda.to_device(np.ascontiguousarray(image))
    pooled=np.array(np.ones(final_dims), dtype=np.uint8)
    dpooled=cuda.to_device(pooled)

    max_pool_unit[(bpgy, bpgx), (tpb, tpb)](dimage, pooling, stride, dpooled)

    dpooled.to_host()
    return pooled


def main():
    pass


if __name__ == '__main__':
    main()
