def mut_normal(self, prob, mu=0.0, sigma=0.05):
    masks = self.rng.random(self.Px.shape) < prob
    for mask, xi in zip(masks, self.Px):
        if any(mask):
            xi[mask] += self.rng.normal(loc=mu, scale=sigma, size=sum(mask))
    return self.Px

def mut_uniform(self, prob):
    masks = self.rng.random(self.Px.shape) < prob
    for mask, xi in zip(masks, self.Px):
        if any(mask):
            xi[mask] = self.rng.uniform(self.lower_bounds[mask], self.upper_bounds[mask], sum(mask))
    return self.Px

def mut_cauchy(self, prob, scaling=0.1):
    masks = self.rng.random(self.Px.shape) < prob
    for mask, xi in zip(masks, self.Px):
        if any(mask):
            xi[mask] += self.rng.standard_cauchy(sum(mask))*scaling
    return self.Px

def mut_none(self, prob):
    return self.Px

mut_dict = {"cauchy"  : mut_cauchy, 
            "normal"  : mut_normal, 
            "uniform" : mut_uniform,
            "none"    : mut_none}