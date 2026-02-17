"""Visualize one representative 3D facet per orbit class."""
import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from vertices import make_qp, make_qm
from prismatoid import group_facets, build_dual_graph
from minkowski import classify_facets
from quotient import (
    make_sigma_plus, group_vertex_sets,
    compute_orbits, quotient_graph,
)


def facet_vertex_indices(hull, s2g):
    """Map each group to its set of vertex indices."""
    gverts = defaultdict(set)
    for i, simplex in enumerate(hull.simplices):
        gverts[s2g[i]].update(simplex)
    return gverts


def project_to_3d(pts):
    """Project 4D points to 3D via SVD."""
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:3].T


def facet_2d_faces(pts_3d):
    """Get polygonal 2D faces of a 3D convex hull.

    Returns list of (vertex_indices, centroid) per face.
    Groups triangular simplices by supporting plane.
    """
    hull = ConvexHull(pts_3d)
    eqs = hull.equations
    norms = np.linalg.norm(eqs[:, :3], axis=1, keepdims=True)
    eqs_n = eqs / norms
    # Group simplices by plane
    plane_groups = defaultdict(set)
    for i, eq in enumerate(eqs_n):
        key = tuple(np.round(eq, decimals=5))
        plane_groups[key].update(hull.simplices[i])
    faces = []
    for verts in plane_groups.values():
        verts = list(verts)
        center = pts_3d[verts].mean(axis=0)
        # Sort vertices by angle around centroid
        rel = pts_3d[verts] - center
        # Project onto the face plane
        _, _, Vt = np.linalg.svd(rel, full_matrices=False)
        proj = rel @ Vt[:2].T
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        order = np.argsort(angles)
        faces.append(([verts[i] for i in order], center))
    return faces


def match_face_to_neighbor(
    face_verts_global, group_id, adj, gverts, f2o
):
    """Find which orbit a 2D face borders."""
    fset = set(face_verts_global)
    best_g = None
    best_overlap = 0
    for nbr in adj.get(group_id, set()):
        overlap = len(fset & gverts[nbr])
        if overlap > best_overlap:
            best_overlap = overlap
            best_g = nbr
    if best_g is not None:
        return f2o[best_g]
    return None


def plot_facet(ax, pts_3d, faces, face_orbits, orbit_id, bd):
    """Plot one 3D facet with colored, labeled faces."""
    cmap = plt.cm.tab10
    for (verts, center), orb in zip(faces, face_orbits):
        poly = [pts_3d[verts]]
        color = cmap(orb % 10) if orb is not None else (0.8,) * 3
        fc = list(color[:3]) + [0.5]
        ec = list(color[:3]) + [1.0]
        collection = Poly3DCollection(
            poly, facecolor=fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_collection3d(collection)
        label = str(orb) if orb is not None else "?"
        ax.text(
            center[0], center[1], center[2], label,
            ha='center', va='center', fontsize=7,
            fontweight='bold',
        )
    # Set equal aspect
    all_pts = pts_3d
    mid = all_pts.mean(axis=0)
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_title(
        f"Orbit {orbit_id}\nbidim {bd}",
        fontsize=9
    )
    ax.set_axis_off()


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
    gverts = facet_vertex_indices(hull, s2g)

    n_orbits = len(orbits)
    fig = plt.figure(figsize=(20, 8))
    cols = 5
    rows = (n_orbits + cols - 1) // cols

    for oi, orb in enumerate(orbits):
        rep = next(iter(orb))
        bd = bidims[rep]
        vidx = sorted(gverts[rep])
        pts_4d = V[vidx].astype(float)
        pts_3d = project_to_3d(pts_4d)
        # Map local indices back to global
        local_to_global = {
            loc: glob for loc, glob in enumerate(vidx)
        }
        faces = facet_2d_faces(pts_3d)
        face_orbits = []
        for verts_local, _ in faces:
            verts_global = {local_to_global[v] for v in verts_local}
            orb_id = match_face_to_neighbor(
                verts_global, rep, adj, gverts, f2o
            )
            face_orbits.append(orb_id)

        ax = fig.add_subplot(rows, cols, oi + 1, projection='3d')
        plot_facet(ax, pts_3d, faces, face_orbits, oi, bd)

    plt.tight_layout()
    plt.savefig("facets.png", dpi=150, bbox_inches='tight')
    print("Saved facets.png")
    plt.show()


if __name__ == "__main__":
    main()
