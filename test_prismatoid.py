"""Tests for the Santos prismatoid construction."""
import numpy as np
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import (
    build_prismatoid, group_facets,
    build_dual_graph, bfs_eccentricity,
)


def test_vertex_counts():
    Qp = make_qp()
    Qm = make_qm(Qp)
    assert len(Qp) == 24
    assert len(Qm) == 24


def test_prismatoid_is_union_not_minkowski():
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = build_prismatoid(Qp, Qm)
    assert len(V) == 48
    assert np.all(V[:24, 4] == 1)
    assert np.all(V[24:, 4] == -1)


def test_prismatoid_facet_count():
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = build_prismatoid(Qp, Qm)
    hull = ConvexHull(V)
    _, n_facets = group_facets(hull)
    assert n_facets == 322


def test_width_is_six():
    """Distance from A (Q+) to L (Q-) is 6 > d = 5."""
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = build_prismatoid(Qp, Qm)
    hull = ConvexHull(V)
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    # Identify A (x5 >= 0, i.e. -x5 <= 0) and L (x5 <= 0)
    eqs = hull.equations
    norms = np.linalg.norm(
        eqs[:, :-1], axis=1, keepdims=True
    )
    eqs_n = eqs / norms
    # A: normal ~ (0,0,0,0,-1), offset ~ 1 => -x5 <= 1
    # L: normal ~ (0,0,0,0,+1), offset ~ 1 => +x5 <= 1
    facet_A = facet_L = None
    for i in range(len(eqs)):
        g = s2g[i]
        n_vec = eqs_n[i, :5]
        if np.allclose(n_vec, [0, 0, 0, 0, -1], atol=1e-4):
            facet_A = g
        if np.allclose(n_vec, [0, 0, 0, 0, 1], atol=1e-4):
            facet_L = g
    assert facet_A is not None and facet_L is not None
    ecc = bfs_eccentricity(adj, facet_A, n_facets)
    # BFS from A: check distance to L
    from collections import deque
    dist = [-1] * n_facets
    dist[facet_A] = 0
    q = deque([facet_A])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    assert dist[facet_L] == 6


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
