import numpy as np
import pandas as pd
import random
from tqdm import tqdm

def calculate_iv_fast(feature, target, n_bins=10):
    # Simplified IV calc for speed
    try:
        # Binning
        bins = pd.qcut(feature, n_bins, duplicates='drop', retbins=True)[1]
        if len(bins) < 2: return 0.0
        
        # Digitize
        inds = np.digitize(feature, bins)
        
        df = pd.DataFrame({'bin': inds, 'y': target})
        stats = df.groupby('bin')['y'].agg(['count', 'sum'])
        stats['good'] = stats['sum']
        stats['bad'] = stats['count'] - stats['sum']
        
        total_good = stats['good'].sum()
        total_bad = stats['bad'].sum()
        
        if total_good == 0 or total_bad == 0: return 0.0
        
        dist_good = stats['good'] / total_good
        dist_bad = stats['bad'] / total_bad
        
        woe = np.log((dist_good + 1e-5) / (dist_bad + 1e-5))
        iv = (dist_good - dist_bad) * woe
        return iv.sum()
    except:
        return 0.0

class AutoFeatureGA:
    def __init__(self, features, labels, pop_size=50, generations=10):
        self.features = features
        self.labels = labels
        self.pop_size = pop_size
        self.generations = generations
        self.n_features = features.shape[1]
        self.ops = ['+', '-', '*', '/']
        
    def random_individual(self):
        # Gene: (idx1, idx2, op)
        return (
            random.randint(0, self.n_features-1),
            random.randint(0, self.n_features-1),
            random.choice(self.ops)
        )
        
    def evaluate(self, individual):
        idx1, idx2, op = individual
        f1 = self.features[:, idx1]
        f2 = self.features[:, idx2]
        
        if op == '+': res = f1 + f2
        elif op == '-': res = f1 - f2
        elif op == '*': res = f1 * f2
        elif op == '/': res = f1 / (f2 + 1e-5)
        
        return calculate_iv_fast(res, self.labels)
        
    def run(self):
        population = [self.random_individual() for _ in range(self.pop_size)]
        
        best_overall = None
        best_score = -1
        
        for g in range(self.generations):
            scores = []
            for ind in population:
                score = self.evaluate(ind)
                scores.append((ind, score))
                
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores[0][1] > best_score:
                best_score = scores[0][1]
                best_overall = scores[0][0]
                
            print(f"Generation {g}: Best IV = {scores[0][1]:.4f} | Gene: {scores[0][0]}")
            
            # Selection (Top 50%)
            survivors = [x[0] for x in scores[:self.pop_size//2]]
            
            # Crossover & Mutation
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                
                # Crossover
                child = (p1[0], p2[1], p1[2]) # Mix indices and op
                
                # Mutation
                if random.random() < 0.2:
                    child = self.random_individual()
                    
                new_pop.append(child)
            
            population = new_pop
            
        return best_overall, best_score

if __name__ == "__main__":
    # Mock data test
    N = 1000
    D = 10
    X = np.random.randn(N, D)
    # Make dim 0 and 1 useful
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    ga = AutoFeatureGA(X, y, pop_size=20, generations=5)
    best_gene, best_iv = ga.run()
    print(f"Final Result: {best_gene} with IV={best_iv:.4f}")
