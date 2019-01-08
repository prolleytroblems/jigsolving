import numpy as np
import pickle
import random



class Permutation(object):

    def __init__(self, objects, fitness_func, length=None):
        #position n has objects[n]
        if not(objects is -1):
            if not(isinstance(objects, tuple)):
                raise Exception("Permutation must be in the form of a tuple")
            self.objects=objects
        else:
            self.random_init(length)
        self.fitness=fitness_func()

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
        if random.random()<cross_p:
        raise NotImplementedError()

    def rand_single_mutate(self, pos, mutate_p):
        if random.random()<mutate_p:
            second = random.choice(list(range(len(self.objects))))
            temp = self.objects[pos]
            self.objects[pos] = self.objects[second]
            self.objects[second] = temp

    def rand_mutate(self, mutate_p):
        for pos in range(len(self.objects)):
            self.rand_single_mutate(pos, mutate_p)



class Generation(list):

    def __init__(self, chromossomes=None, cross_p=0.3, mutate_p=0.01):
        if chromossomes:
            super().__init__(chromossomes)
        else:
            super().__init__()
        random.seed(hash(os.urandom(4)))

    def next_generation(self, size=len(self), elitism=0.1):
        next_gen=Generation(self.best(round(elitism*len(self))))
        for _ in range(len(self)-len(next_gen)):
            if selection="tournament":
                pair=[self.tournament(), self.tournament()]
            elif selection="roulette":
                pair=[self.roulette(), self.roulette()]

            pair=pair[0].rand_crossover(pair[1])
            for chromossome in pair:
                next_gen.append(chromossome.rand_mutate(mutate_p))
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

    def __init__(self):
        pass
