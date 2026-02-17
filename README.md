# Hirsch counterexample computations

Computational verification and visualization of Santos's
counterexample to the Hirsch conjecture
([arXiv:1006.2814](https://arxiv.org/abs/1006.2814)).

## What this computes

Santos constructs a 5-dimensional prismatoid with 48 vertices
whose dual graph has diameter 6 > d = 5, disproving the
Hirsch conjecture.

The key objects are two 4-dimensional polytopes Q+ and Q-
(each with 24 vertices). The code:

1. **Prismatoid**: Lifts Q+ to x5=+1 and Q- to x5=-1, takes
   the convex hull. Verifies 48 vertices, 322 facets, and
   dual graph diameter 7 (width 6).

2. **Minkowski sum** Q+ + Q-: Computes the Minkowski sum in
   R^4, classifies facets by bidimension (i, j), and computes
   distances in the dual graph.

3. **Quotient graph**: Partitions the 272 Minkowski sum facets
   into orbits under the order-32 symmetry group Sigma+.
   Uses exact integer vertex-set matching.

4. **Visualization**: Projects one representative 3D facet per
   orbit, labels 2D faces by adjacent orbit class. Exports to
   JSON for an interactive Three.js viewer (`index.html`).

## Files

| File | Purpose |
|------|---------|
| `vertices.py` | Q+ and Q- vertex definitions (48, 32, 28 variants) |
| `prismatoid.py` | 5-prismatoid construction and dual graph |
| `minkowski.py` | Minkowski sum, bidimension classification |
| `quotient.py` | Sigma+ symmetry group, orbit decomposition |
| `visualize_facets.py` | Matplotlib 3D facet plots |
| `visualize_facets2.py` | JSON export for Three.js viewer |
| `index.html` | Interactive Three.js visualization |

## Usage

```bash
# Run prismatoid verification
python3 -m pytest test_prismatoid.py -v

# Generate facets.json and run visualization tests
python3 -m pytest test_visualize.py -v

# Generate facets.json manually
python3 visualize_facets2.py

# Open interactive viewer
open index.html
```

## Dependencies

- Python 3, NumPy, SciPy, matplotlib, pytest
