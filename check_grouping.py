"""Diagnose facet grouping in the Minkowski sum."""
import numpy as np
from scipy.spatial import ConvexHull
from vertices import make_qp, make_qm
from prismatoid import group_facets

Qp = make_qp()
Qm = make_qm(Qp)

hp = ConvexHull(Qp)
hm = ConvexHull(Qm)
_, fp = group_facets(hp)
_, fm = group_facets(hm)
print(f"Q+ facets: {fp}, Q- facets: {fm}")

V = np.unique(
    np.array([u + v for u in Qp for v in Qm]), axis=0
)
hull = ConvexHull(V)
for dec in [4, 5, 6, 7, 8]:
    _, nf = group_facets(hull, decimals=dec)
    print(f"decimals={dec}: {nf} facets")
