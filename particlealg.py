from particle import Particle
import numpy as np
from copy import copy


class ParticleOptimizer:

    def __init__(self, n_dims, n_particles, fitness, value_ranges=None, **kwargs):

        self.particles=[]
        pos = np.random.random((n_particles, n_dims))
        spd = np.random.random((n_particles, n_dims))
        if value_ranges:
            value_ranges=np.asarray(value_ranges)
            pos = pos*(value_ranges[:,1]-value_ranges[:,0])+value_ranges[:,0]
        for i in range(n_particles):
            self.particles.append(Particle(pos[i], spd[i], fitness, **kwargs))
        self.gbest=None
        self._uptodate=False

    def get_best(self):
        a=0
        if self._uptodate:
            best=self._gbest
        else:
            best=[self.particles[0].get()]
            a=1
        for i in range(a, len(self.particles)):
            temp=self.particles[i].get()
            if temp[0]>best[0][0]:
                best=[temp]
            elif temp[0]==best[0][0]:
                best.append(temp)
        return best

    def gbestprop():
        doc = "The gbest property."
        def fget(self):
            if self._uptodate:
                return self._gbest
            else:
                best=self.get_best()
                if not(self._gbest) or best[0][0]>self._gbest[0][0]:
                    self._gbest=best
                else:
                    pass
                self._uptodate=True
                return self._gbest
        def fset(self, value):
            if value==None:
                self._gbest=None
            else:
                raise Exception("Dont mess with gbest manually")
        def fdel(self):
            del self._gbest
        return locals()
    gbest = property(**gbestprop())

    def step(self):
        for particle in self.particles:
            particle.step(self.gbest[0][1])
        self._uptodate=False
