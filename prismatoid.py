"""Compute Santos's 5-prismatoid and its dual graph."""
import numpy as np
from collections import defaultdict, deque
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm


def build_prismatoid(Qp, Qm):
    """Lift Q+ to x5=+1, Q- to x5=-1, return their union."""
    Qp_lift = np.hstack([Qp, np.ones((len(Qp), 1))])
    Qm_lift = np.hstack([Qm, -np.ones((len(Qm), 1))])
    return np.vstack([Qp_lift, Qm_lift])


def group_facets(hull, decimals=6):
    """Group simplicial facets sharing a hyperplane.

    Returns (simplex_to_group, num_groups).
    """
    eqs = hull.equations
    norms = np.linalg.norm(
        eqs[:, :-1], axis=1, keepdims=True
    )
    eqs_normed = eqs / norms
    groups = {}
    simplex_to_group = np.empty(len(eqs), dtype=int)
    for i, eq in enumerate(eqs_normed):
        key = tuple(np.round(eq, decimals=decimals))
        if key not in groups:
            groups[key] = len(groups)
        simplex_to_group[i] = groups[key]
    return simplex_to_group, len(groups)


def build_dual_graph(hull, simplex_to_group):
    """Facet adjacency via Qhull neighbor info."""
    adj = defaultdict(set)
    for i, neighbors in enumerate(hull.neighbors):
        gi = simplex_to_group[i]
        for j in neighbors:
            gj = simplex_to_group[j]
            if gi != gj:
                adj[gi].add(gj)
    return adj


def bfs_eccentricity(adj, start, n):
    """Max distance from start in BFS."""
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return max(dist)


def diameter(adj, n):
    """Diameter of the dual graph by BFS from every node."""
    return max(
        bfs_eccentricity(adj, s, n) for s in range(n)
    )


if __name__ == "__main__":
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = build_prismatoid(Qp, Qm)
    hull = ConvexHull(V)
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    diam = diameter(adj, n_facets)
    print(f"Vertices: {len(V)}")
    print(f"Facets: {n_facets}")
    print(f"Dual graph diameter: {diam}")
