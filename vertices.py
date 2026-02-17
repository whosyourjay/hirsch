"""Santos's Q+ and Q- polytopes in R^4 (24 vertices each)."""
import itertools
import numpy as np


def make_qp():
    return make_qp48()

def make_qp48():
    """Q+ vertices from Santos's counterexample."""
    Qp = []
    for s in [-1, 1]:
        Qp.append([s * 18, 0, 0, 0])
        Qp.append([0, s * 18, 0, 0])
        Qp.append([0, 0, s * 45, 0])
        Qp.append([0, 0, 0, s * 45])
    for s1, s2 in itertools.product([-1, 1], repeat=2):
        Qp.append([s1 * 15, s2 * 15, 0, 0])
        Qp.append([0, 0, s1 * 30, s2 * 30])
        Qp.append([0, s1 * 10, s2 * 40, 0])
        Qp.append([s1 * 10, 0, 0, s2 * 40])
    return np.array(Qp)

def make_qp32():
    """Q+ vertices from simplified counterexampl."""
    Qp = []
    for s in [-1, 1]:
        Qp.append([s * 72, 0, 0, 0])
        Qp.append([0, s * 45, 0, 0])
        Qp.append([0, 0, s * 120, 0])
        Qp.append([0, 0, 0, s * 120])
    for s1, s2 in itertools.product([-1, 1], repeat=2):
        Qp.append([0, s1 * 20, 0, s2 * 100])
        Qp.append([0, 0, s1 * 72, s2 * 72])
    return np.array(Qp)

def make_qp28():
    """Q+ vertices from simplified counterexampl."""
    Qp = []
    for s in [-1, 1]:
        Qp.append([s * 18, 0, 0, 0])
        Qp.append([0, 0, s * 30, 0])
        Qp.append([0, 0, 0, s * 30])
    for s1, s2 in itertools.product([-1, 1], repeat=2):
        Qp.append([0, s1 * 5, 0, s2 * 25])
        Qp.append([0, 0, s1 * 18, s2 * 18])
    return np.array(Qp)


def make_qm(Qp):
    return Qp[:, [2, 3, 1, 0]]
