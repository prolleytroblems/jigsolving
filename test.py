from discretedarwin import *
import numpy as np
from datetime import datetime

table= np.zeros((5,5))
table[0,3]=1
table[1,1]=1
table[2,0]=1
table[3,4]=1
table[4,2]=1

table = np.random.random((10,10))

pop=[]
for _ in range(1000):
    pop.append(PositionPerm(objects=-1, value_table=table, length=10))


def fitness(chrom):
    return chrom.fitness

gen=Generation(pop)

for i in range(100):
    out=gen.best(10)
print(out, fitness(out[0]))


pair=out[0:2]
print(pair)
print(pair[0].rand_crossover(pair[1], 1))


start=datetime.now()
gen=gen.next_gen()
print(datetime.now()-start)

for i in range(100):
    out=gen.best(10)
print(out, fitness(out[0]))
