from discretedarwin import *
import numpy as np
from datetime import datetime
from copy import deepcopy

def one_run(mp, xp, el):
    gen=Generation(deepcopy(pop), cross_p=xp, mutate_p=mp, elitism=el)
    for i in range(200):
        gen=gen.next_generation()
    return(gen.best(1)[0].fitness)


def test(tries, func):
    tot=0
    for _ in range(tries):
        tot+=func()
    return tot/tries


table = np.random.random((20,20))
for i in range(20):
    table[i,i]=1
table=table**4
for j in range(10):
    pop=[]
    for _ in range(100):
        pop.append(PositionPerm(objects=-1, value_table=table, length=20))

    gen=Generation(deepcopy(pop), cross_p=0.1, mutate_p=0.04, elitism=0.05)
    for i in range(600):
        gen=gen.next_generation()
    print(gen.best(1)[0].fitness)


"""
fout=open("out.txt", "w")
header= "mutate_p, cross_p, elitism, pop_size, performance"
fout.write(header)
print(header)

for mp in range(3, 8, 1):
    for xp in range(10, 50, 10):
        for el in range(4, 8, 1):
            func=lambda : one_run(mp/100, xp/100, el/100)
            result=test(10, func)
            output=str(mp/100)+", "+str(xp/100)+", "+str(el/100)+", "+str(result)+"\n"
            print(output)
            fout.write(output)
fout.close()"""
