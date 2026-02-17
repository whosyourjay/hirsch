"""Facet adjacency of Q+ + Q- modulo Sigma+."""
import numpy as np
import itertools
from collections import defaultdict
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import group_facets, build_dual_graph
from minkowski import classify_facets


def make_sigma_plus():
    """Generate the 32 elements of Sigma+."""
    I4 = np.eye(4)
    swap = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)
    matrices = []
    for signs in itertools.product([-1, 1], repeat=4):
        S = np.diag(signs)
        matrices.append(I4 @ S)
        matrices.append(swap @ S)
    return matrices


def group_vertex_sets(hull, s2g, V):
    """Map each grouped facet to frozenset of integer vertex coords."""
    gverts = defaultdict(set)
    for i, simplex in enumerate(hull.simplices):
        gverts[s2g[i]].update(simplex)
    V_int = np.rint(V).astype(int)
    return {
        g: frozenset(tuple(V_int[v]) for v in verts)
        for g, verts in gverts.items()
    }


def compute_orbits(sigma, gvsets, n_facets):
    """Compute orbits using exact integer vertex-set matching."""
    vset_to_group = {vs: g for g, vs in gvsets.items()}
    visited = set()
    orbits = []
    facet_to_orbit = {}
    for g in range(n_facets):
        if g in visited:
            continue
        orbit = {g}
        for M in sigma:
            M_int = np.rint(M).astype(int)
            transformed = frozenset(
                tuple(M_int @ np.array(v))
                for v in gvsets[g]
            )
            target = vset_to_group.get(transformed)
            if target is not None:
                orbit.add(target)
        orbits.append(orbit)
        for f in orbit:
            visited.add(f)
            facet_to_orbit[f] = len(orbits) - 1
    return orbits, facet_to_orbit


def quotient_graph(adj, facet_to_orbit, n_orbits):
    """Build adjacency graph on orbits."""
    edges = set()
    for f, nbrs in adj.items():
        of = facet_to_orbit[f]
        for n in nbrs:
            on = facet_to_orbit[n]
            if of != on:
                e = (min(of, on), max(of, on))
                edges.add(e)
    qadj = defaultdict(set)
    for a, b in edges:
        qadj[a].add(b)
        qadj[b].add(a)
    return qadj, edges


if __name__ == "__main__":
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = np.unique(
        np.array([u + v for u in Qp for v in Qm]), axis=0
    )
    hull = ConvexHull(V)
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    bidims = classify_facets(hull, s2g, n_facets, Qp, Qm)
    sigma = make_sigma_plus()
    gvsets = group_vertex_sets(hull, s2g, V)

    orbits, f2o = compute_orbits(sigma, gvsets, n_facets)
    qadj, edges = quotient_graph(adj, f2o, len(orbits))

    print(f"Orbits: {len(orbits)}")
    print(f"Edges: {len(edges)}")
    for i, orb in enumerate(orbits):
        rep = next(iter(orb))
        bd = bidims[rep]
        print(
            f"  Orbit {i}: size {len(orb)}, "
            f"bidim {bd}, "
            f"degree {len(qadj.get(i, set()))}"
        )
    print("\nEdge list:")
    for a, b in sorted(edges):
        print(f"  {a} -- {b}")
