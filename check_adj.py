"""Check for adjacency bugs in Minkowski sum hull."""
import numpy as np
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import group_facets

Qp = make_qp()
Qm = make_qm(Qp)
V = np.unique(
    np.array([u + v for u in Qp for v in Qm]), axis=0
)
hull = ConvexHull(V)
s2g, n_facets = group_facets(hull)

# Check for -1 in neighbors
neg = (hull.neighbors == -1).sum()
print(f"Negative neighbor entries: {neg}")

# Check vertex-based adjacency for group 3
# Get vertices per group
from collections import defaultdict
group_verts = defaultdict(set)
for i, simplex in enumerate(hull.simplices):
    g = s2g[i]
    group_verts[g].update(simplex)

# Count adjacencies via shared vertices (>= 3)
g3_nbrs_simplex = set()
for i, neighbors in enumerate(hull.neighbors):
    if s2g[i] == 3:
        for j in neighbors:
            if s2g[j] != 3:
                g3_nbrs_simplex.add(s2g[j])

g3_nbrs_vertex = set()
v3 = group_verts[3]
for g, verts in group_verts.items():
    if g != 3 and len(v3 & verts) >= 3:
        g3_nbrs_vertex.add(g)

print(f"Group 3 neighbors (simplex): {sorted(g3_nbrs_simplex)}")
print(f"Group 3 neighbors (vertex):  {sorted(g3_nbrs_vertex)}")
print(f"Group 3 vertex count: {len(v3)}")
