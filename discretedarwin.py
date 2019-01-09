import numpy as np
import pickle
import random
from os import urandom
from copy import deepcopy


class Permutation(object):

    def __init__(self, objects, fitness_func, length=None):
        #position n has objects[n]
        if not(objects is -1):
            if not(isinstance(objects, tuple)):
                raise Exception("Permutation must be in the form of a tuple")
            self.objects=objects
        else:
            self.random_init(length)
        self.fitness_func=fitness_func
        self.fitness=self.fitness_func()

    def refresh(self):
        self.fitness=self.fitness_func()

    def random_init(self, length):
        self.objects=list(range(length))
        random.shuffle(self.objects)

    def __str__(self):
        return str(self.objects)

    def __repr__(self):
        return repr(self.objects)

    def crossover(self, other):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()


class PositionPerm(Permutation):

    def __init__(self, objects, value_table, length=None):
        self.table=value_table
        super().__init__(objects, self.position_fitness, length)

    def position_fitness(self):
        score=0
        for object, position in enumerate(self.objects):
            score+=self.table[object, position]
        return score

    def rand_crossover(self, other, cross_p):
        new_pair = [deepcopy(self), deepcopy(other)]
        if random.random()<cross_p:
            split_loc = int(random.random()*len(self.objects))
            new_pair[0].crossover(other, split_loc)
            new_pair[1].crossover(self, split_loc)
        return new_pair

    def crossover(self, other, split_loc):
        new_order = []

        if split_loc>len(self.objects)//2:
            to_find = self.objects[split_loc:]
            for gene in other.objects:
                if gene in to_find:
                    new_order.append(gene)
        else:
            to_find = self.objects[:split_loc]
            for gene in other.objects:
                if not(gene in to_find):
                    new_order.append(gene)
        self.objects[split_loc:] = new_order
        return self

    def rand_single_mutate(self, pos, mutate_p):
        if random.random()<mutate_p:
            second = int(random.random()*len(self.objects))
            temp = self.objects[pos]
            self.objects[pos] = self.objects[second]
            self.objects[second] = temp

    def rand_mutate(self, mutate_p):
        new_perm = deepcopy(self)
        for pos in range(len(self.objects)):
            new_perm.rand_single_mutate(pos, mutate_p)
        return new_perm



class Generation(list):

    def __init__(self, chromossomes=None, cross_p=0.13, mutate_p=0.04, elitism=0.05, selection="tournament"):
        self.params={"selection":selection,
                    "elitism":elitism,
                    "cross_p":cross_p,
                    "mutate_p":mutate_p}
        if chromossomes:
            super().__init__(chromossomes)
        else:
            super().__init__()
        random.seed(hash(urandom(4)))

    def update_fitness(self):
        for chromossome in self:
            chromossome.refresh()

    def next_generation(self, size=None):
        "size rounds to next largest pair"
        if not(isinstance(size, int)):
            size=len(self)
        next_gen=Generation(self.best(round(self.params["elitism"]*len(self))), **self.params)
        left = size-len(next_gen)
        while len(next_gen)<size:
            if self.params["selection"]=="tournament":
                pair=[self.tournament(), self.tournament()]
            elif self.params["selection"]=="roulette":
                pair=[self.roulette(), self.roulette()]

            pair=pair[0].rand_crossover(pair[1], self.params["cross_p"])
            for chromossome in pair:
                chro=chromossome.rand_mutate(self.params["mutate_p"])
                next_gen.append(chro)
        if len(next_gen)>size:
            next_gen.pop()
        next_gen.update_fitness()
        return next_gen

    def roulette(self):
        raise NotImplementedError()

    def tournament(self):
        pair=random.sample(self, 2)
        if pair[0].fitness>pair[1].fitness:
            return pair[0]
        else:
            return pair[1]

    def best(self, n, population=None):
        if n<=0:
            return None
        if not(population):
            population=self
        if n>len(population):
            raise ValueError("n must not be larger than the length of population.")

        best=sorted(population[0:n], key=lambda x:x.fitness, reverse=True)

        for chromossome in population[n:]:
            podium=-1
            for position in range(1, len(best)+1):
                if chromossome.fitness>best[-position].fitness:
                    if len(best)<position+1:
                        podium = position
                        break
                    else:
                        continue
                else:
                    if position>1:
                        podium = position-1
                        break
                    else:
                        break
            if podium>0:
                best.insert(-podium, chromossome)
                if len(best)>n:
                    best.pop()
        return best


class DiscreteDarwin(object):

    def __init__(self, table, pop_size, len, **kwargs):
        assert table.shape==(len, len)
        self.len=len
        self.table=table
        self.pop_size=pop_size
        self.initialize_pop( **kwargs)

    def initialize_pop(self, **kwargs):
        self.gen=Generation(**kwargs)
        for _ in range(self.pop_size):
            self.gen.append(PositionPerm(-1, self.table, self.len))

    def advance(self):
        self.gen=self.gen.next_generation()

    def run(self, generations):
        for _ in range(generations):
            self.advance()
            print(self.best().fitness)
        return self.gen.best(1)[0]

    def best(self):
        return self.gen.best(1)[0]
