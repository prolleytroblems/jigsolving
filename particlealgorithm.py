import numpy as np
import random


class Particle:

    def __init__(self, position, speed, fitness,
            lrate=(0.1,0.1), randsigma=(0.1,0.1), tinterval=0.1):
        position=np.asarray(position)
        speed=np.asarray(speed)
        assert len(position.shape)==1
        assert len(speed.shape)==1
        assert position.shape==speed.shape

        self.postion=position
        self.speed=speed
        self.fitness=fitness
        self.pbest=[self.position)]
        self.lrate=np.asarray(accelerate)
        self.randigma=np.asarray(randsigma)
        self.tinterval=tinterval

    def get(self):
        return (self.fitness(self.position), self.position)

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
        dists=distance(bests)
        randomness=np.random.normal(0, self.randsigma, (2))
        diffs=bests-np.asarray((self.position, self.position))
        accel=np.dot(np.transpose(diffs), self.lrate*randomness)
        return accel

    def step(self, gbest):
        self.accelerate(self.calcaccel(gbest))
        self.movestep()
        score=self.fitness(self.position):
        if score>self.pbest:
            self.pbest=[self.position]
        elif score==self.pbest:
            self.pbest.append(self.position)

        if score>gbest:
            return (gbest, 0)
        elif score==gbest:
            return (gbest, 1)
        else:
            return None
