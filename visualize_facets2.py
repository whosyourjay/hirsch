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
    """Compute (col, row) positions via BFS from (3,0) orbit.

    Assigns top/middle/bottom rows based on which parent
    each node was reached from.
    """
    from collections import deque
    # Find the (3,0) orbit
    start = None
    for oid in range(n_orbits):
        rep = next(iter(orbits[oid]))
        if bidims[rep] == (3, 0):
            start = oid
            break
    dist = [-1] * n_orbits
    dist[start] = 0
    q = deque([start])
    layers = defaultdict(list)
    layers[0].append(start)
    while q:
        u = q.popleft()
        for v in qadj.get(u, set()):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                layers[dist[v]].append(v)
                q.append(v)
    # Assign rows: for 1-node layers, center (row=0).
    # For 2-node layers: top(-1) and bottom(+1).
    # For 3-node layers: assign based on connections
    # to previous layer's top/bottom nodes.
    pos = {}
    prev_top = None
    prev_bot = None
    for col in sorted(layers.keys()):
        nodes = layers[col]
        if len(nodes) == 1:
            pos[nodes[0]] = (col, 0)
            prev_top = prev_bot = nodes[0]
        elif len(nodes) == 2:
            # Assign arbitrarily, first=top
            pos[nodes[0]] = (col, -1)
            pos[nodes[1]] = (col, 1)
            prev_top, prev_bot = nodes[0], nodes[1]
        elif len(nodes) == 3:
            top_n = bot_n = mid_n = None
            for n in nodes:
                nbrs = qadj.get(n, set())
                has_top = prev_top in nbrs
                has_bot = prev_bot in nbrs
                if has_top and not has_bot:
                    top_n = n
                elif has_bot and not has_top:
                    bot_n = n
                else:
                    mid_n = n
            pos[top_n] = (col, -1)
            pos[bot_n] = (col, 1)
            pos[mid_n] = (col, 0)
            prev_top, prev_bot = top_n, bot_n
    return pos


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
    s2g, n_facets = group_facets(hull)
    adj = build_dual_graph(hull, s2g)
    bidims = classify_facets(hull, s2g, n_facets, Qp, Qm)
    sigma = make_sigma_plus()
    gvsets = group_vertex_sets(hull, s2g, V)
    orbits, f2o = compute_orbits(sigma, gvsets, n_facets)
    qadj, _edges = quotient_graph(adj, f2o, len(orbits))
    layout = bfs_layout(
        qadj, len(orbits), bidims, orbits
    )

    gverts = defaultdict(set)
    for i, simp in enumerate(hull.simplices):
        gverts[s2g[i]].update(simp)

    out = []
    for oid, orb in enumerate(orbits):
        rep = next(iter(orb))
        vidx = sorted(gverts[rep])
        loc2glob = {loc: gl for loc, gl in enumerate(vidx)}
        pts4 = V[vidx].astype(float)
        pts3 = project_to_3d(pts4)

        # Normalize to unit bounding box
        span = pts3.max(axis=0) - pts3.min(axis=0)
        scale = max(span) if max(span) > 0 else 1
        pts3 = pts3 / scale

        polys = facet_polygons(pts3)
        faces = []
        face_labels = []
        for vs_local, _uv in polys:
            vs_global = {loc2glob[v] for v in vs_local}
            nbr_oid = match_face_to_neighbor(
                vs_global, rep, adj, gverts, f2o
            )
            lbl = LETTERS[nbr_oid] if nbr_oid is not None else "?"
            faces.append([int(v) for v in vs_local])
            face_labels.append(lbl)

        # Schlegel diagram for flat view
        face_idx_list = [f for f in faces]
        pts2d = schlegel_2d(pts3, face_idx_list)
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
            "label": LETTERS[oid],
            "bidim": [int(bd[0]), int(bd[1])],
            "pos": [col, row],
            "vertices": [
                [round(float(x), 5) for x in row]
                for row in pts3
            ],
            "faces": faces,
            "face_labels": face_labels,
            "flat_faces": flat_faces,
        })

    with open("facets.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote facets.json: {len(out)} orbits")


if __name__ == "__main__":
    main()
