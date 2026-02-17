"""Check which (3,0) facets are at distance 5 from (0,3)."""
import numpy as np
from collections import deque
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import group_facets, build_dual_graph
from minkowski import classify_facets, bfs_distances

Qp = make_qp()
Qm = make_qm(Qp)
V = np.unique(
    np.array([u + v for u in Qp for v in Qm]), axis=0
)
hull = ConvexHull(V)
s2g, n_facets = group_facets(hull)
adj = build_dual_graph(hull, s2g)
bidims = classify_facets(hull, s2g, n_facets, Qp, Qm)

src = [g for g, b in bidims.items() if b == (3, 0)]
tgt = set(g for g, b in bidims.items() if b == (0, 3))

for s in src:
    dist = bfs_distances(adj, s, n_facets)
    min_d = min(dist[t] for t in tgt)
    if min_d >= 5:
        print(f"Facet {s}: min dist to (0,3) = {min_d}")
        print(f"  degree = {len(adj[s])}")
        nbrs = adj[s]
        nbr_types = [bidims[n] for n in nbrs]
        print(f"  neighbor bidims: {nbr_types}")
