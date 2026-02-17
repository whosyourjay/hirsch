"""Minkowski sum Q+ + Q- in R^4."""
import numpy as np
from collections import deque
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import group_facets, build_dual_graph, diameter


def affine_dim(points, tol=1e-8):
    """Affine dimension of a point set."""
    if len(points) <= 1:
        return 0
    diffs = points[1:] - points[0]
    return np.linalg.matrix_rank(diffs, tol=tol)


def bidimension(normal, Qp, Qm, tol=1e-6):
    """Bidimension (i, j) of a Minkowski sum facet."""
    dp = Qp @ normal
    dq = Qm @ normal
    Fp = Qp[dp >= dp.max() - tol]
    Fm = Qm[dq >= dq.max() - tol]
    return affine_dim(Fp), affine_dim(Fm)


def classify_facets(hull, s2g, n_facets, Qp, Qm):
    """Map each grouped facet to its bidimension."""
    # Pick one normal per group
    group_normal = {}
    eqs = hull.equations
    for i, g in enumerate(s2g):
        if g not in group_normal:
            group_normal[g] = eqs[i, :4]
    return {
        g: bidimension(n, Qp, Qm)
        for g, n in group_normal.items()
    }


def bfs_distances(adj, source, n):
    """BFS distances from source to all nodes."""
    dist = [-1] * n
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def min_max_distance(adj, n_facets, bidims, src_type, tgt_type):
    """max over src_type facets of min over tgt_type facets of dist."""
    sources = [g for g, b in bidims.items() if b == src_type]
    targets = [g for g, b in bidims.items() if b == tgt_type]
    result = float('inf')
    for s in sources:
        dist = bfs_distances(adj, s, n_facets)
        min_d = min(dist[t] for t in targets)
        result = min(result, min_d)
    return result


if __name__ == "__main__":
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = np.unique(
        np.array([u + v for u in Qp for v in Qm]),
        axis=0,
    )
    hull = ConvexHull(V)
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    bidims = classify_facets(hull, s2g, n_facets, Qp, Qm)

    from collections import Counter
    counts = Counter(bidims.values())
    print(f"Vertices: {len(hull.vertices)}")
    print(f"Facets: {n_facets}")
    print(f"Bidimension counts: {dict(sorted(counts.items()))}")
    print(f"Dual graph diameter: {diameter(adj, n_facets)}")
    d = min_max_distance(adj, n_facets, bidims, (3, 0), (0, 3))
    print(f"min_(3,0) min_(0,3) dist: {d}")
