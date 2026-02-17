"""Tests for facet visualization data."""
import json
import subprocess
import pytest


@pytest.fixture(scope="module")
def facets_data():
    subprocess.run(
        ["python3", "visualize_facets2.py"], check=True
    )
    with open("facets.json") as f:
        return json.load(f)


def test_orbit_count(facets_data):
    assert len(facets_data) == 10


def test_labels_are_letters(facets_data):
    expected = set("ABCDEFGHIJ")
    labels = {d["label"] for d in facets_data}
    assert labels == expected


def test_no_degenerate_flat_faces(facets_data):
    for d in facets_data:
        for ff in d["flat_faces"]:
            pts = ff["polygon"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            assert w > 0.001 and h > 0.001, (
                f"Degenerate face in orbit {d['label']}"
            )


def test_face_labels_reference_valid_orbits(facets_data):
    valid = {d["label"] for d in facets_data}
    for d in facets_data:
        for lbl in d["face_labels"]:
            assert lbl in valid, (
                f"Invalid label {lbl} in orbit {d['label']}"
            )


def test_no_np_int64_in_json():
    with open("facets.json") as f:
        raw = f.read()
    assert "int64" not in raw
