import numpy as np
from numba import cuda
from image_obj import Solution
from utils import param_check
from datetime import datetime

DEFAULTS={"debug_mode":False, "pool_method":"avg"}

def pool(images_or_solution, pooling, stride, **params):
    params=param_check(params, DEFAULTS)

    if params["debug_mode"]==True:
        start=datetime.now()

    if isinstance(images_or_solution, np.ndarray) or isinstance(images_or_solution, list):
        assert len(np.array(images_or_solution).shape)==4
        pooled=[]
        for image in images_or_solution:
            pooled.append(pool_image(image, pooling, stride, **params))
        if params["debug_mode"]==True:
            print("Pooling images, "+params["pool_method"]+": "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")
        return np.array(pooled, dtype=np.uint8)

    elif isinstance(images_or_solution, Solution):
        dims=images_or_solution.shape
        images_or_solution = images_or_solution.arrays
        pooled=[]
        for image in images_or_solution:
            pooled.append(pool_image(image, pooling, stride, **params))
        if params["debug_mode"]==True:
            print("Pooling solution, "+params["pool_method"]+": "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")
        return Solution(np.array(pooled, dtype=np.uint8), dims)

    else: raise TypeError("images_or_solution must be an ndarray or Solution instance")

def pool_image(image, pooling, stride, pool_method="avg", **params):
    """Apply pooling to a single image. \n
        image    ndarray \n
        pooling  tuple of int of len=2 \n
        stride   tuple of int of len=2 \n """

    if not(pooling[0]>=stride[0] and pooling[1]>=stride[1]): raise TypeError("Pooling and stride inputs should be ndarray-like")
    if not(len(image.shape)==3): raise TypeError("Image must be a 3D ndarray.")

    pooling=np.array(pooling, dtype=np.uint8)
    stride=np.array(stride, dtype=np.uint8)

    for axis in range(2):
        image=add_padding(image, axis, pooling, stride, "end")

    final_dims=((image.shape[0]-pooling[0]+stride[0])//stride[0], (image.shape[1]-pooling[1]+stride[1])//stride[1], 3)
    tpb=16
    bpgy=(final_dims[0]-1)//tpb+1
    bpgx=(final_dims[1]-1)//tpb+1

    dimage=cuda.to_device(np.ascontiguousarray(image))
    pooled=np.array(np.zeros(final_dims), dtype=np.uint8)
    dpooled=cuda.to_device(pooled)

    func = {"max": max_pool_unit, "avg": avg_pool_unit}

    func[pool_method][(bpgy, bpgx), (tpb, tpb)](dimage, pooling, stride, dpooled)

    dpooled.to_host()

    return pooled

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

@cuda.jit
def avg_pool_unit(image, pooling, stride, pooled):
    y, x=cuda.grid(2)

    if y>pooled.shape[0] or x>pooled.shape[1]:
        return
    window=image[y*stride[1]:y*stride[1]+pooling[1], x*stride[0]:x*stride[0]+pooling[0], :]

    pooled[y,x,0]=0
    pooled[y,x,1]=0
    pooled[y,x,2]=0

    for i in range(pooling[0]):
        for j in range(pooling[1]):
            for c in range(3):
                pooled[y,x,c]+=window[i,j,c]
    for c in range(3):
        window[i,j,c]=window[i,j,c]/(pooling[0]/pooling[1])

def add_padding(image, axis, pooling, stride, side="end"):
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
            image=add_padding(image, axis, pooling, stride, side="start")
        elif side=="start":
            image=np.concatenate((np.zeros(((image.shape[0]-1)*coeff[0]+1,
                    (image.shape[1]-1)*coeff[1]+1, 3), dtype=np.uint8), image), axis=axis)
            image=add_padding(image, axis, pooling, stride, side="end")
        return image
