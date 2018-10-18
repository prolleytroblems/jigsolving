import numpy as np
import random
from copy import copy


class Particle:

    def __init__(self, position, speed, fitness,
            lrate=(1,1), randsigma=(0.1,0.1), tinterval=0.1):
        position=np.asarray(position)
        speed=np.asarray(speed)
        assert len(position.shape)==1
        assert len(speed.shape)==1
        assert position.shape==speed.shape

        self.position=position
        self.speed=speed
        self.fitness=fitness
        self.pbest=[copy(self.position)]
        self.lrate=np.asarray(lrate)
        self.randsigma=np.asarray(randsigma)
        self.tinterval=tinterval

    def get(self):
        return (self.fitness(self.position), copy(self.position))

    def accelerate(self, acceleration):
        self.speed+=acceleration

    def movestep(self):
        self.position+=self.speed*self.tinterval

    def distance(self, positions):
        if len(positions.shape)==1:
            return np.apply_along_axis(lambda x: x**2, 0, self.position-position)
        else:
            return np.apply_along_axis(distance, 0, positions)

    def calcaccel(self, gbest):
        pbest=random.choice(self.pbest)
        bests=np.asarray((gbest,pbest))
        randomness=np.random.normal(0, self.randsigma, (2))
        diffs=bests-np.asarray((self.position, self.position))
        accel=np.dot(np.transpose(diffs), self.lrate*randomness)
        return accel

    def step(self, gbest):
        self.accelerate(self.calcaccel(gbest))
        self.movestep()
        pbestfit=self.fitness(self.pbest[0])
        score=self.fitness(self.position)
        if score>pbestfit:
            self.pbest=[copy(self.position)]
        elif score==pbestfit:
            self.pbest.append(self.position)

        if score>gbest[1]:
            return 1
        elif score==gbest[1]:
            return 2
        else:
            return 0
