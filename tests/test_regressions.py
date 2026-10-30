"""
Regression tests for bugs found during the maintenance review.
"""

import os
import tempfile
import unittest

import numpy as np

import tetgenpy


def cube_points():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )


CUBE_FACES = [
    [1, 0, 2, 3],
    [0, 1, 5, 4],
    [2, 0, 4, 6],
    [1, 3, 7, 5],
    [3, 2, 6, 7],
    [4, 5, 7, 6],
]


class PointAttributeTest(unittest.TestCase):
    def test_point_attributes_are_copied_from_attributes(self):
        points = cube_points()
        attrs = np.arange(16, dtype=float).reshape(8, 2) + 100.0

        tio = tetgenpy.TetgenIO()
        tio.setup_points(points, attrs, [])

        np.testing.assert_array_equal(tio.pointattributes(), attrs)
        np.testing.assert_array_equal(tio.points(), points)


class TetrahedronAttributeTest(unittest.TestCase):
    def test_attributes_keep_fractional_values(self):
        plc = tetgenpy.PLC()
        plc.add_points(cube_points())
        plc.add_facets(CUBE_FACES)
        plc.add_regions([[0.1, 0.1, 0.1, 1.5, 1.0]])

        out = tetgenpy.tetrahedralize("QA", plc.to_tetgenio())
        attrs = out.tetrahedronattributes()

        self.assertEqual(attrs.dtype, np.float64)
        np.testing.assert_allclose(attrs, 1.5)


class FacetsWithHolesWithoutMarkersTest(unittest.TestCase):
    def test_h_facets_without_facet_markers(self):
        # used to write through a null facetmarkerlist
        v = np.vstack([cube_points(), cube_points() * [0.4, 1, 0.4] + [0.3, 0, 0.3]])
        front = [[[0, 1, 5, 4], [8, 9, 13, 12]], [[0.5, 0.0, 0.5]], 1]
        back = [[[2, 3, 7, 6], [10, 11, 15, 14]], [[0.5, 1.0, 0.5]], 2]
        sides = [
            [1, 3, 7, 5],
            [5, 7, 6, 4],
            [4, 6, 2, 0],
            [0, 2, 3, 1],
            [9, 11, 15, 13],
            [13, 15, 14, 12],
            [12, 14, 10, 8],
            [8, 10, 11, 9],
        ]

        tio = tetgenpy.TetgenIO()
        tio.setup_plc(v, sides, h_facets=[front, back])
        out = tetgenpy.tetrahedralize("Q", tio)

        self.assertGreater(len(out.tetrahedra()), 0)


class SaveEdgesTest(unittest.TestCase):
    def test_save_edges_writes_edge_file(self):
        plc = tetgenpy.PLC()
        plc.add_points(cube_points())
        out = tetgenpy.tetrahedralize("Qe", plc.to_tetgenio())

        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cube")
            out.save_edges(base)
            self.assertTrue(os.path.exists(base + ".edge"))
            self.assertFalse(os.path.exists(base + ".neigh"))


if __name__ == "__main__":
    unittest.main()
