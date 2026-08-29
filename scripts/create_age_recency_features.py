import numpy as np
from itertools import combinations

# construct first/last time of each edge (u,v) in simplices
def build_pair_first_last(simplices):

    first = {}
    last = {}
    for time, simp in sorted(simplices, key=lambda x: x[0]):  # ascending order by time
        for u, v in combinations(sorted(simp), 2):
            key = (u, v)
            if key not in first:
                first[key] = time
            last[key] = time
    return first, last

# age: lifetime of triangle from birth to T_end (T_end-t_first)
# recency: freshness of triangle from last activity to T_end (T_end-t_last)

def calculate_age_recency(triangle, pair_first, pair_last, T_end):
    """
    Important Definitions:
    first: the time when the open triangle is formed 
            the max of the first appearance time of its three edges
    last: the time when the open triangle is last active in observed time window
            the max of the last appearance time of its three edges
    """
    a, b, c = triangle
    tfs = [pair_first.get(p) for p in ((a, b), (a, c), (b, c))]
    tls = [pair_last.get(p) for p in ((a, b), (a, c), (b, c))]
    if any(t is None for t in tfs):      
        return 0.0, 0.0
    t_first = max(tfs)
    t_last = max(tls)
    age = float(np.log1p(max(0, T_end - t_first))) # ln(1+x): avoid long tail effect
    recency = float(np.log1p(max(0, T_end - t_last)))
    return age, recency


# usage
def compute_for_candidates(simplices, trg_open):
    """return a list of (age, recency) for each triangle in trg_open"""
    pair_first, pair_last = build_pair_first_last(simplices)
    T_end = max(t for t, _ in simplices) # T_end = max(self.simplex_to_time.values())
    return [calculate_age_recency(tri, pair_first, pair_last, T_end)
            for tri in trg_open]