from numba import cuda,jit
from datetime import datetime
import numpy as np
import cv2
from image_obj import *
from utils import *
from discretedarwin import DiscreteDarwin
from pooling import *


DEFAULTS={"debug_mode":False, "threshold":None, "iterator_mode":False, "method":"xcorr", "id_only":True}

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

@cuda.jit(device=True)
def compare_xcorr_color(colora, colorb, means):
    c=(colora-means[0])*(colorb-means[1])
    return c

@cuda.jit
def gcompare_xcorr(imga, imgb, means, C, weights):
    y, x = cuda.grid(2)
    if y>=imga.shape[0] or x>=imga.shape[1]:
        return
    for i in range(3):
        C[y,x,i]=compare_xcorr_color(imga[y,x,i], imgb[y,x,i], means[:,i]) * weights[i]

def compare(dimga, dimgb, **params):
    tpb = 16
    bpgy = (dimga.shape[0]-1)//tpb+1
    bpgx = (dimga.shape[1]-1)//tpb+1

    C=np.array(np.zeros(dimga.shape[0:2]), dtype=np.float32)

    gcompare[(bpgy, bpgx), (tpb, tpb)](dimga, dimgb, C)

    return 1-np.sum(C)/(dimga.shape[0]*dimga.shape[1]*441.7)

def statistics(array):
    means=[]
    stds=[]
    for i in range(3):
        means.append(array[:,:,i].mean())
        stds.append(array[:,:,i].std())
    return (means, stds)

def compare_xcorr(imga, imgb, dimga, dimgb, weights=np.array((1,1,1), dtype=np.float32), **params):

    means_a, stds_a = statistics(imga)
    means_b, stds_b = statistics(imgb)
    N=imga.shape[0]*imga.shape[1]
    means=np.array((means_a, means_b))

    tpb = 16
    bpgy = (dimga.shape[0]-1)//tpb+1
    bpgx = (dimga.shape[1]-1)//tpb+1

    C=np.array(np.zeros(dimga.shape), dtype=np.float32)
    gcompare_xcorr[(bpgy, bpgx), (tpb, tpb)](dimga, dimgb, means, C, weights)

    xcorr=[]
    for i in range(3):
        div=( (N-1) * stds_a[i] * stds_b[i])
        if div!=0:
            xcorr.append(np.sum( C[:,:,i]) / div)
        else:
            xcorr.append(np.sum( C[:,:,i]) / ( div + 0.00001 ))

    return sum(xcorr)/3

def locate_one_piece(dpiece, solution, **params):
    """Will only receive preprocessed device arrays!"""
    params=param_check(params, DEFAULTS)
    raise Exception("Deprecated")
    if params["debug_mode"]==True:
        start=datetime.now()

    piece=dpiece.copy_to_host()


    max_resemblance=[0, None, 0, 1]
    #maximum resemblance, location index, second max resemblance(for debugging), min resemblance (for debugging)

    for i in range(solution.darrays.shape[0]):
        if solution.availability[i]==True:
            solutionpiece=solution.darrays[i].copy_to_host()
            resemblance=compare(dpiece, solution.darrays[i], **params)

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
        print("Piece index"+str(params["index"])+
                ", Max res.: "+ str(max_resemblance[0])+
                ", 2nd max res.: "+ str(max_resemblance[2])+
                ", Min res.: "+ str(max_resemblance[3])+
                ", "+ "Runtime: "+ str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return solution.locations[max_resemblance[1]]

def preprocess_pieces_old(pieces, solution, pooling=None, **params):
    params=param_check(params, DEFAULTS)

    if params["debug_mode"]==True:
        start=datetime.now()

    if not(isinstance(pieces, np.ndarray) and isinstance(solution, Solution)):raise TypeError("Wrong object type!")
    if not(len(pieces.shape)==4 and len(solution.arrays.shape)==4): raise Exception("Incorrect array shape!")
    if pieces.shape[0] != solution.arrays.shape[0]: raise Exception("Number of pieces don't match!")

    if pieces.shape != solution.arrays.shape:
        print(pieces.shape, solution.arrays.shape)
        print("Piece shape mismatch!")
        pieces=resize_batch(pieces, (solution.arrays[0].shape[0:2][::-1]), **params)

    if pooling != None and pooling != 1:
        pieces=pool(pieces, (pooling, pooling), (pooling, pooling), **params)
        solution=pool(solution, (pooling, pooling), (pooling, pooling), **params)

    if params["debug_mode"]==True:
        print("Preprocessing: "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return (pieces, solution)

def preprocess_pieces(images, solution, pooling=None, **params):
    params=param_check(params, DEFAULTS)

    if params["debug_mode"]==True:
        start=datetime.now()
    target_shape=solution.arrays.shape[1:]
    target_size=(target_shape[1], target_shape[0])
    pieces=[]
    for array in images:
        if array.shape != target_shape:
            pieces.append(cv2.resize(array, target_size))
        else:
            pieces.append(array)
    pieces=np.array(pieces)

    if pooling != None and pooling != 1:
        pieces=pool(pieces, (pooling, pooling), (pooling, pooling), **params)
        solution=pool(solution, (pooling, pooling), (pooling, pooling), **params)

    if params["debug_mode"]==True:
        print("Preprocessing: "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return (pieces, solution)

def find_match(dto_match, dpieces, availability=None, mask=None, **params):
    """Will only receive preprocessed device arrays! \n
        Returns the index of best matching piece from array pieces."""

    params=param_check(params, DEFAULTS)

    if params["debug_mode"]==True:
        start=datetime.now()
    if availability==None:
        av_array=[True]*dpieces.shape[0]
    else:
        assert all(e==True or e==False for e in availability)
        av_array=availability
    if mask==None:
        mask=[True]*dpieces.shape[0]

    max_resemblance=[-1, None, -1, 1]
    #maximum resemblance, match index, second max resemblance(for debugging), min resemblance (for debugging)
    solutionpiece=dto_match.copy_to_host()
    for i in range(dpieces.shape[0]):
        if not(mask[i]):
            continue
        if params["method"]=="square error":
            resemblance=compare(dpieces[i], dto_match, **params)
        elif params["method"]=="xcorr":
            piece=dpieces[i].copy_to_host()
            resemblance=compare_xcorr(piece, solutionpiece, dpieces[i], dto_match, **params)
        else:
            raise Exception("Invalid method selection:", params["method"])

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

    if availability!=None:
        availability[max_resemblance[1]]=False

    if params["debug_mode"]==True:

        print(" {0:<13}  {1:<13.5f}  {2:<13.5f}  {3:<13.5f}  {4:<13.2f}".format(
                params["index"], max_resemblance[0], max_resemblance[2], max_resemblance[3],
                float((datetime.now()-start).seconds*1000)+float((datetime.now()-start).microseconds)/1000))

    return max_resemblance[1]

def resize_batch(pieces, size, **params):
    resized=[]
    for piece in pieces:
        resized.append(cv2.resize(piece, size))
    return np.array(resized)

def locate_pieces_iter(pieces, solution, pooling=None, **params):
    params=param_check(params, DEFAULTS)

    p_pieces, p_solution = preprocess_pieces(pieces, solution, pooling, **params)
    dpieces = cuda.to_device(np.ascontiguousarray(p_pieces))

    if params["debug_mode"]:
        print("{0:<15s}{1:<15s}{2:<15s}{3:<15s}{4:<15s}".format("Piece index", "Max res.", "2nd max res.", "Min res.", "Runtime (ms)"))

    for i in range(len(p_solution.locations)):
        if params["debug_mode"]:
            params["index"]=i

        index=find_match(p_solution.darrays[i], dpieces, **params)
        if not(params["id_only"]):
            pieces[index].slot=p_solution.locations[i]
            yield(piece[index])
        else:
            yield((piece[index].id, p_solution.locations[i]))

def locate_pieces(pieces, solution, pooling=None, **params):
    params=param_check(params, DEFAULTS)

    p_pieces, p_solution = preprocess_pieces(pieces.mass_get("image"), solution, pooling, **params)
    dpieces = cuda.to_device(np.ascontiguousarray(p_pieces))

    out=[]
    piece_mask=[True]*len(pieces)

    if params["debug_mode"]:
        print(params)
        print("{0:<15s}{1:<15s}{2:<15s}{3:<15s}{4:<15s}".format("Piece index", "Max res.", "2nd max res.", "Min res.", "Runtime (ms)"))


    piece_locations=[-1]*len(pieces)
    for i in range(len(p_solution.slots)):
        if params["debug_mode"]:
            params["index"]=i

        index=find_match(p_solution.darrays[i], dpieces, availability=p_solution.availability, mask=piece_mask, **params)
        piece_mask[index]=False
        piece_locations[index]=p_solution.slots[i]

    assert not(any(i is -1 for i in piece_locations))
    if not(params["id_only"]):
        pieces.mass_set("slot", piece_locations)
        return pieces
    else:
        return(list(zip(pieces.mass_get("id"), piece_locations)))

def full_solve(pieces, solution, pooling=None, **params):
    params=param_check(params, DEFAULTS)

    if params["debug_mode"]:
        start=datetime.now()

    if params["method"]=="genalg(xcorr)":
        solved=genalg_solve(pieces, solution, pooling=pooling, **params)
    else:
        if not(params["iterator_mode"]):
            if not(params["id_only"]):
                solved=locate_pieces(pieces, solution, pooling=pooling, **params)
                solved=PieceCollection(solved, find_dims())
            else:
                print(len(pieces), solution, pooling, params)
                solved=locate_pieces(pieces, solution, pooling=pooling, **params)
        else:
            raise NotImplementedError("iterator solve not properly implemented")
            solved=locate_pieces_iter(pieces, solution, pooling=pooling, **params)

    if params["debug_mode"]:
        if params["iterator_mode"]==True:
            print("Iterator mode: True")
        else:
            print("Iterator mode: False")
        print("Solving: "+str((datetime.now()-start).seconds*1000+float((datetime.now()-start).microseconds)/1000)+" ms")

    return solved

def sort_pieces(located_pieces, dims):
    sorted_pieces=[[0 for column in range(dims[1])] for row in range(dims[0])]
    for piece in located_pieces:
        sorted_pieces[piece.location[0]][piece.location[1]]=piece

    return [piece for row in sorted_pieces for piece in row]

def get_valuearray(pieces, solutionpieces, dpieces, dsolution, **params):
    valuearray=np.zeros((len(dpieces), len(dpieces)))
    for i in range(len(dsolution)):
        for j in range(len(dpieces)):
            #valuearray[i, j] = compare(dsolution[i], dpieces[j], decoding="sort", **params)**2
            valuearray[i, j] = compare_xcorr(solutionpieces[i], pieces[j], dsolution[i], dpieces[j], **params)
    np.savetxt("valuearray.csv", valuearray)
    return valuearray

def reduce_search(valuearray):
    raise NotImplementedError()
    exclusion_mask_rows=np.ones((valuearray.shape[0]), dtype=boolean)
    exclusion_mask_cols=np.ones((valuearray.shape[0]), dtype=boolean)
    for i, row in enumerate(valuearray):
        if np.argmax(valuearray[:, np.argmax(row)]) == i:
            exclusion_mask[i] = False
        else:
            pass

    return (new_valuearray, partial_solution)

def genalg_solve(pieces, solution, pooling=None, **params):
    p_pieces, p_solution = preprocess_pieces(pieces.mass_get("image"), solution, pooling, **params)
    dpieces = cuda.to_device(np.ascontiguousarray(p_pieces))

    dsolution = p_solution.darrays
    valuearray = get_valuearray(p_pieces, p_solution.arrays, dpieces, dsolution)

    #valuearray, partial_solution = reduce_search(valuearray)

    optimizer = DiscreteDarwin(valuearray, 100, valuearray.shape[0] )
    optimizer.run(200)
    permutation=optimizer.best()

    """permutation=iter(permutation.objects)
    for position, object in enumerate(partial_solution):
        if object == -1:
            partial_solution[position] == next(permutation)
    permutation=partial_solution"""

    try:
        next(permutation)
        raise Exception("damn")
    except StopIteration:
        pass

    print(permutation)

    piece_locations=[-1]*len(pieces)
    for location_index, piece_index in enumerate(permutation.objects):
        piece_locations[piece_index]=p_solution.slots[location_index]

    if not(params["id_only"]):
        pieces.mass_set("slot", piece_locations)
        return pieces
    else:
        return(list(zip(pieces.mass_get("id"), piece_locations)))


def main():
    pass

if __name__ == '__main__':
    main()
