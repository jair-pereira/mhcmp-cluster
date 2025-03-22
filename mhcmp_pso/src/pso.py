import numpy as np
from typing import Type, Sequence, Annotated

class PSO():
    def __init__(self,
                 rng: Type[np.random.default_rng],
                 n: int, 
                 w: float, 
                 c1: float, 
                 c2: float,
                 dimension: int, 
                 lower_bounds: Annotated[Sequence[float], "Length[dimension]"], 
                 upper_bounds: Annotated[Sequence[float], "Length[dimension]"]):
        
        ## rng ##
        self.rng = rng

        ## meta-parameters ##
        self.n   = n
        self.w   = w
        self.c1  = c1
        self.c2  = c2
        
        ## optimization properties ##
        self.dimension    = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
    
    ## OPERATORS ##
    def initialize_solutions(self):
        return np.array([self.rng.uniform(self.lower_bounds, self.upper_bounds, self.dimension) for _ in range(self.n)])
    
    def initialize_velocity(self):
        return np.zeros((self.n, self.dimension))
    
    def update_gbest(self):
        current_best = np.argmin(self.Pf)
        if self.gbest_f >= self.Pf[current_best]:
            self.gbest_x = self.Px[current_best].copy()
            self.gbest_f = self.Pf[current_best]
    
    def update_pbest(self):
        for i in range(self.n):
            if self.Pbest_f[i] >= self.Pf[i]:
                self.Pbest_x[i] = self.Px[i].copy()
                self.Pbest_f[i] = self.Pf[i]
    
    def update_velocity(self):
        R1 = self.rng.random((self.n, self.dimension))
        R2 = self.rng.random((self.n, self.dimension))
        self.Pv = np.array([self.w*self.Pv[i] 
                            + self.c1*R1[i]*(self.Pbest_x[i] - self.Px[i]) 
                            + self.c2*R2[i]*(self.gbest_x    - self.Px[i]) for i in range(self.n)])
        
    def update_position(self):
        self.Px = self.Px + self.Pv
    
    def mutation(self):
        return

    def repair(self):
        self.Px = np.array([np.clip(self.Px[i], self.lower_bounds, self.upper_bounds) 
                            for i in range(self.n)])

    ## ASK & TELL ##
    def ask0(self):
        ## data structures ##
        self.Px = self.initialize_solutions() # list of candidate solutions
        self.Pv = self.initialize_velocity()  # list of velocities
        self.Pf = np.array([np.inf]*self.n)   # list of objective values
        
        ## global best ##
        self.gbest_x = None
        self.gbest_f = np.inf
        
        ## personal best ##
        self.Pbest_x = np.array([x.copy() for x in self.Px])
        self.Pbest_f = np.array([np.inf]*self.n)
        
        return self.Px

    def ask(self):
        self.update_velocity()
        self.update_position()
        self.mutation()
        self.repair()
        
        return self.Px
    
    def tell(self, Pf: Annotated[Sequence[float], "Length[dimension]"]):
        self.Pf = Pf
        
        self.update_gbest()
        self.update_pbest()

