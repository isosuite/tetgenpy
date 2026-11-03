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
        v = np.vstack(
            [cube_points(), cube_points() * [0.4, 1, 0.4] + [0.3, 0, 0.3]]
        )
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


class NonContiguousAndDtypeRobustnessTest(unittest.TestCase):
    def test_non_contiguous_and_fortran_arrays(self):
        # 16 points, sliced with step 2 (non-contiguous)
        pts = np.vstack([cube_points(), cube_points() + 2.0])
        sliced_pts = pts[::2]
        self.assertFalse(sliced_pts.flags.c_contiguous)

        tio = tetgenpy.TetgenIO()
        tio.setup_points(sliced_pts)
        out = tetgenpy.tetrahedralize("Q", tio)
        self.assertGreater(len(out.points()), 0)

        # Fortran contiguous array
        f_pts = np.asfortranarray(cube_points())
        self.assertFalse(f_pts.flags.c_contiguous)
        tio_f = tetgenpy.TetgenIO()
        tio_f.setup_points(f_pts)
        out_f = tetgenpy.tetrahedralize("Q", tio_f)
        self.assertGreater(len(out_f.points()), 0)

    def test_int64_and_float32_conversion(self):
        # numpy on 64-bit platforms defaults to int64
        pts_f32 = cube_points().astype(np.float32)
        tio = tetgenpy.TetgenIO()
        tio.setup_points(pts_f32)
        out = tetgenpy.tetrahedralize("Q", tio)

        in_tets = out.tetrahedra()
        self.assertGreater(len(in_tets), 0)

        # Pass int64 tetrahedra to setup_tetmesh
        tets_i64 = in_tets.astype(np.int64)
        tio2 = tetgenpy.TetgenIO()
        tio2.setup_tetmesh(out.points(), tets_i64)
        out2 = tetgenpy.tetrahedralize("rQ", tio2)
        self.assertGreaterEqual(len(out2.tetrahedra()), len(in_tets))


class ThreadingAndGilReleaseTest(unittest.TestCase):
    def test_concurrent_tetrahedralize(self):
        import concurrent.futures

        def mesh_cube(scale):
            pts = cube_points() * scale
            tio = tetgenpy.TetgenIO()
            tio.setup_points(pts)
            out = tetgenpy.tetrahedralize("Q", tio)
            return len(out.tetrahedra())

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(mesh_cube, i + 1.0) for i in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 8)
        for count in results:
            self.assertGreater(count, 0)


class DefaultArgumentsTest(unittest.TestCase):
    def test_setup_with_defaults(self):
        # Test calling setup_points without optional arguments
        tio = tetgenpy.TetgenIO()
        tio.setup_points(cube_points())
        out = tetgenpy.tetrahedralize("Q", tio)
        self.assertGreater(len(out.tetrahedra()), 0)

        # Test calling setup_plc without optional arguments
        tio_plc = tetgenpy.TetgenIO()
        tio_plc.setup_plc(cube_points(), CUBE_FACES)
        out_plc = tetgenpy.tetrahedralize("Q", tio_plc)
        self.assertGreater(len(out_plc.tetrahedra()), 0)


class ErrorHandlingTest(unittest.TestCase):
    def test_invalid_shape_exception_message(self):
        tio = tetgenpy.TetgenIO()
        # Shape mismatch in column count
        with self.assertRaises(RuntimeError) as ctx:
            tio.setup_points(np.zeros((5, 2)))
        self.assertIn("shape mismatch in dimension [ 1 ]", str(ctx.exception))

        # Dim mismatch (1D instead of 2D)
        with self.assertRaises(RuntimeError) as ctx2:
            tio.setup_points(np.zeros(5))
        self.assertIn("dim mismatch", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
