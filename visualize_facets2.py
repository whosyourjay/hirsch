"""Export facet geometry + labels as JSON for Three.js."""
import json
import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull

from vertices import make_qp, make_qm
from prismatoid import group_facets, build_dual_graph
from minkowski import classify_facets
from quotient import (
    make_sigma_plus, group_vertex_sets,
    compute_orbits, quotient_graph,
)

LETTERS = "BCDEFGHIJKLMNOPQRSTUVWXYZ"


def project_to_3d(pts):
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:3].T


def facet_polygons(pts3):
    """Return list of (local_vertex_indices, 2d_coords) per face."""
    hull = ConvexHull(pts3)
    eqs = hull.equations
    norms = np.linalg.norm(eqs[:, :3], axis=1, keepdims=True)
    eqs_n = eqs / norms
    planes = defaultdict(set)
    for i, eq in enumerate(eqs_n):
        key = tuple(np.round(eq, 6))
        planes[key].update(hull.simplices[i])
    polys = []
    for vs in planes.values():
        vs = list(vs)
        c = pts3[vs].mean(axis=0)
        rel = pts3[vs] - c
        _, _, Vt = np.linalg.svd(rel, full_matrices=False)
        uv = rel @ Vt[:2].T
        ang = np.arctan2(uv[:, 1], uv[:, 0])
        order = np.argsort(ang)
        sorted_vs = [vs[i] for i in order]
        sorted_uv = uv[order].tolist()
        polys.append((sorted_vs, sorted_uv))
    return polys


def schlegel_2d(pts3, face_indices_list):
    """Schlegel diagram: project 3D polyhedron to 2D.

    Projects from just outside the largest face,
    making all other faces visible inside it.
    Returns pts2d array (n_verts x 2).
    """
    # Pick largest face as outer boundary
    outer_i = max(
        range(len(face_indices_list)),
        key=lambda i: len(face_indices_list[i])
    )
    outer = face_indices_list[outer_i]
    p0, p1, p2 = pts3[outer[0]], pts3[outer[1]], pts3[outer[2]]
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n)

    # Ensure outward-pointing
    center_poly = pts3.mean(axis=0)
    center_face = pts3[outer].mean(axis=0)
    if np.dot(n, center_face - center_poly) < 0:
        n = -n

    # Projection center just outside the face
    diam = np.linalg.norm(
        pts3.max(axis=0) - pts3.min(axis=0)
    )
    proj_center = center_face + 0.3 * diam * n

    # 2D basis on the face plane
    e1 = p1 - p0
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)

    # Project each vertex
    pts2d = np.zeros((len(pts3), 2))
    for i, v in enumerate(pts3):
        ray = v - proj_center
        denom = np.dot(n, ray)
        if abs(denom) < 1e-12:
            pts2d[i] = [np.dot(v - center_face, e1),
                        np.dot(v - center_face, e2)]
        else:
            t = np.dot(n, center_face - proj_center) / denom
            p = proj_center + t * ray
            pts2d[i] = [np.dot(p - center_face, e1),
                        np.dot(p - center_face, e2)]
    return pts2d


def bfs_layout(qadj, n_orbits, bidims, orbits):
    """Compute (col, row) positions via BFS from a (3,0) orbit.

    Spreads each BFS layer evenly along the row axis,
    sorted by average neighbor row in the previous layer.
    """
    from collections import deque
    starts = []
    for oid in range(n_orbits):
        rep = next(iter(orbits[oid]))
        if bidims[rep] == (3, 0):
            starts.append(oid)
    dist = [-1] * n_orbits
    q = deque()
    layers = defaultdict(list)
    for s in starts:
        dist[s] = 0
        q.append(s)
        layers[0].append(s)
    while q:
        u = q.popleft()
        for v in qadj.get(u, set()):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                layers[dist[v]].append(v)
                q.append(v)
    pos = {}
    bfs_order = []
    for col in sorted(layers.keys()):
        nodes = layers[col]
        k = len(nodes)
        if col > 0 and k > 1:
            def sort_key(n):
                nbrs = qadj.get(n, set())
                rows = [pos[m][1] for m in nbrs
                        if m in pos]
                return sum(rows) / len(rows) if rows else 0
            nodes = sorted(nodes, key=sort_key)
        for i, n in enumerate(nodes):
            row = i - (k - 1) / 2
            pos[n] = (col, row)
        bfs_order.extend(nodes)
    return pos, bfs_order


def match_face_to_neighbor(face_global, gid, adj, gverts, f2o):
    fset = set(face_global)
    best_g, best_n = None, 0
    for nbr in adj.get(gid, set()):
        n = len(fset & gverts[nbr])
        if n > best_n:
            best_n = n
            best_g = nbr
    return f2o[best_g] if best_g is not None else None


def main():
    Qp = make_qp()
    Qm = make_qm(Qp)
    V = np.unique(
        np.array([u + v for u in Qp for v in Qm]), axis=0
    )
    hull = ConvexHull(V)

    # Normalized vertices for rounder 3D projections.
    # Map each integer Minkowski vertex to normalized sum.
    Qp_n = Qp / np.linalg.norm(Qp, axis=1, keepdims=True)
    Qm_n = Qm / np.linalg.norm(Qm, axis=1, keepdims=True)
    int_to_norm = {}
    for u, un in zip(Qp, Qp_n):
        for v, vn in zip(Qm, Qm_n):
            key = tuple(u + v)
            if key not in int_to_norm:
                int_to_norm[key] = un + vn
    V_norm = np.array([int_to_norm[tuple(row)] for row in V])
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    bidims = classify_facets(hull, s2g, n_facets, Qp, Qm)
    sigma = make_sigma_plus()
    gvsets = group_vertex_sets(hull, s2g, V)
    orbits, f2o = compute_orbits(sigma, gvsets, n_facets)
    qadj, _edges = quotient_graph(adj, f2o, len(orbits))
    layout, bfs_order = bfs_layout(
        qadj, len(orbits), bidims, orbits
    )
    # Map old orbit id -> BFS-order letter
    oid_to_letter = {}
    for rank, oid in enumerate(bfs_order):
        oid_to_letter[oid] = LETTERS[rank]

    gverts = defaultdict(set)
    for i, simp in enumerate(hull.simplices):
        gverts[s2g[i]].update(simp)

    out = []
    for oid in bfs_order:
        orb = orbits[oid]
        rep = next(iter(orb))
        vidx = sorted(gverts[rep])
        loc2glob = {loc: gl for loc, gl in enumerate(vidx)}

        # Face structure from integer coords (exact planes)
        pts3_int = project_to_3d(V[vidx].astype(float))
        span = pts3_int.max(axis=0) - pts3_int.min(axis=0)
        scale = max(span) if max(span) > 0 else 1
        pts3_int = pts3_int / scale
        polys = facet_polygons(pts3_int)

        # Normalized coords for display
        pts3 = project_to_3d(V_norm[vidx])
        span = pts3.max(axis=0) - pts3.min(axis=0)
        scale = max(span) if max(span) > 0 else 1
        pts3 = pts3 / scale
        faces = []
        face_labels = []
        for vs_local, _uv in polys:
            vs_global = {loc2glob[v] for v in vs_local}
            nbr_oid = match_face_to_neighbor(
                vs_global, rep, adj, gverts, f2o
            )
            lbl = (oid_to_letter[nbr_oid]
                   if nbr_oid is not None else "?")
            faces.append([int(v) for v in vs_local])
            face_labels.append(lbl)

        # Schlegel diagram for flat view
        pts2d = schlegel_2d(pts3, faces)
        flat_faces = []
        for fi, vs_local in enumerate(faces):
            poly2d = pts2d[vs_local].tolist()
            flat_faces.append({
                "label": face_labels[fi],
                "polygon": [
                    [round(x, 4), round(y, 4)]
                    for x, y in poly2d
                ],
            })

        bd = bidims[rep]
        col, row = layout[oid]
        out.append({
            "label": oid_to_letter[oid],
            "bidim": [int(bd[0]), int(bd[1])],
            "pos": [col, row],
            "vertices": [
                [round(float(x), 5) for x in r]
                for r in pts3
            ],
            "faces": faces,
            "face_labels": face_labels,
            "flat_faces": flat_faces,
        })

    edges = [
        [oid_to_letter[a], oid_to_letter[b]]
        for a, b in _edges
    ]
    result = {"orbits": out, "edges": edges}
    with open("facets.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote facets.json: {len(out)} orbits, "
          f"{len(edges)} edges")


if __name__ == "__main__":
    main()
