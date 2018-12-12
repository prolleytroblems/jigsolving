"""Rectangular box intersection and union library for use with chainer."""
from datetime import datetime
try:
    from chainer import Variable as V
    from chainer.functions import concat as C
    import numpy as np
except Exception as E:
    print(E)
    def V():
        raise Exception("Chainer library missing.")
    C=V



def intersection(boxA, boxB, out="area"):
    """Boxes should be tuples of length 4: x0, y0, width, height."""
    inter_x=intersection1D((boxA[0], boxA[0]+boxA[2]), (boxB[0], boxB[0]+boxB[2]))
    inter_y=intersection1D((boxA[1], boxA[1]+boxA[3]), (boxB[1], boxB[1]+boxB[3]))
    if not(inter_x) or not(inter_y):
        if out=="area":
            return 0
        elif out=="box":
            return None
    elif out=="area":
        return (inter_x[1]-inter_x[0])*(inter_y[1]-inter_y[0])
    elif out=="box":
        return (inter_x[0], inter_y[0], inter_x[1]-inter_x[0], inter_y[1]-inter_y[0])
    else:
        raise Exception("Invalid value for 'out': "+out )

def intersection1D(rangeA, rangeB):
    try:
        start_comp=rangeA[0]>rangeB[0]
        end_comp=rangeA[1]<rangeB[1]
    except:
        tranges=[[],[]]
        for i, range in enumerate((rangeA, rangeB)):
            for dim in (0,1):
                try:
                    tranges[i].append(range[dim].data)
                except:
                    tranges[i].append(range[dim])
        start_comp=tranges[0][0]>tranges[1][0]
        end_comp=tranges[0][1]<tranges[1][1]
    #print(rangeA, rangeB)
    if start_comp:
        start=rangeA[0]
    else:
        start=rangeB[0]
    if end_comp:
        end=rangeA[1]
    else:
        end=rangeB[1]
    intersection=end-start
    #print(intersection)
    #print(type(intersection))
    try:
        pos_intersect=intersection>0
    except:
        pos_intersect=intersection.data>0
    if pos_intersect:
        return (start, end)
    else:
        return None

def union(boxA, boxB=None):
    """Boxes should be tuples of length 4: x0, y0, width, height."""
    if not(boxB):
        return area(boxA)
    areaA=area(boxA)
    areaB=area(boxB)
    return areaA+areaB-intersection(boxA, boxB, out="area")

def IoU(boxA, boxB):
    """Boxes should be tuples of length 4: x0, y0, width, height."""
    intersec = intersection(boxA, boxB, out="area")
    uni = union(boxA, boxB)
    return intersec/uni

def IoA(boxA, boxB):
    """Intersection over area. Boxes should be tuples of length 4: x0, y0, width, height."""
    intersec = intersection(boxA, boxB, out="area")
    areaA=area(boxA)
    areaB=area(boxB)
    return (intersec/areaA, intersec/areaB)

def set_types(obj_list, type, conversion_f):
    out=[]
    for obj in obj_list:
        if not(isinstance(obj, type)):
            #print("converting", obj)
            out.append(conversion_f(obj))
        else:
            out.append(obj)
    return out

def list_IoU(listA, listB, list_type=list):
    #print("start", type(listA[0]), type(listB[0]))
    if list_type==V:
        listA, listB = set_types([listA, listB], V, lambda list: V(np.asarray(list, dtype=np.float32)))
    intersections=[]

    #start=datetime.now()
    for boxA in listA:
        for boxB in listB:
            intbox=intersection(boxA, boxB, out="box")
            if intbox:
                intersections.append(intbox)
    #print("inter", datetime.now()-start)

    #start=datetime.now()
    total_inter=list_union(intersections)
    #print("inter_union", datetime.now()-start)

    #start=datetime.now()
    try:
        big_list=listA+listB
    except:
        big_list=C((listA, listB), axis=0)
    out=total_inter/list_union(big_list)
    #print("big_union", datetime.now()-start)
    return out

def all_combinations(list_items, comb_size):
    if len(list_items)<comb_size:
        return None
    if comb_size==1:
        return [[item] for item in list_items]
    combs=[]
    for i, item in enumerate(list_items):
        sub_combinations=all_combinations(list_items[i+1:], comb_size-1)
        if sub_combinations:
            for combination in sub_combinations:
                combs+=[[item]+combination]
    return combs

def area(box):
    return box[2]*box[3]

def list_union(boxlist):
    interdict={(i,): box for i, box in enumerate(boxlist)}
    sign=1
    total_area=0
    for rank in range(len(boxlist)):
        new_interdict={}
        for index in interdict:
            total_area+=sign*area(interdict[index])
            for i, box in enumerate(boxlist):
                if not(i in index):
                    new_inter=intersection(interdict[index], box, out="box")
                    if new_inter:
                        new_index=list(index)+[i]
                        new_index.sort()
                        new_interdict[tuple(new_index)]=new_inter
        interdict=new_interdict
        sign=sign*(-1)
    return total_area
