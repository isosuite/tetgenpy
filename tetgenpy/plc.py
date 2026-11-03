"""
Class to prepare input Piecewise Linear Complexes for tetgen.
"""

import numpy as np

from tetgenpy import tetgenpy_core as core


def _check_2d_and_shape1(name, array, shape1):
    shape = array.shape
    if len(shape) != 2 or shape[1] != shape1:
        raise ValueError(
            f"{name} should be 2D array-like input with (n, {shape1}) shape."
        )


class PLC:
    """ """

    def __init__(self):
        """
        Initializes all the properties to pass to tetgenio.
        """
        self._points = []
        self._point_attributes = []
        self._point_metrics = []
        self._facets = []
        self._facet_with_holes = []
        self._holes = []
        self._regions = []
        self._facet_markers = []
        self._facet_constraints = []
        self._facet_with_holes_constraints = []
        self._segment_constraints = []

        # self._point_id_offset = 0
        self._facet_id_offset = 0
        self.default_facet_id = -123454321

    def add_points(self, points, point_attributes=None, point_metrics=None):
        """
        Adds points. list, tuple, or np.ndarray.

        Parameters
        ----------
        points: (n, 3)  array-like
        point_attributes: float or (n, m) array-like
        point_metrics: float or (n, 1) array-like

        Returns
        -------
        None
        """
        # keep points as np.ndarray
        points = np.asanyarray(points)

        _check_2d_and_shape1("points", points, 3)

        # append points
        self._points.append(points)

        # append attributes and metrics
        # first, make sure all points or none of them will have attr and mtr
        if point_attributes is None and len(self._point_attributes) > 0:
            raise ValueError(
                "point attributes needs to be set for all points."
            )
        if point_metrics is None and len(self._point_metrics) > 0:
            raise ValueError("point metrics needs to be set for all points.")

        # process point attributes
        if point_attributes is not None:
            if isinstance(point_attributes, list | tuple | np.ndarray):
                point_attributes = np.asanyarray(point_attributes)

            elif isinstance(point_attributes, int | float):
                point_attributes = np.full((len(points), 1), point_attributes)

            # same sized attr
            if len(self._point_attributes) > 0:
                _check_2d_and_shape1(
                    "point_attributes",
                    point_attributes,
                    self._point_attributes[-1].shape(1),
                )

            self._point_attributes.append(point_attributes)

        # process point metrics
        if point_metrics is not None:
            if isinstance(point_metrics, list | tuple | np.ndarray):
                point_metrics = np.asanyarray(point_metrics)

            elif isinstance(point_metrics, int | float):
                point_metrics = np.full((len(points), 1), point_metrics)

            # same sized attr
            if len(self._point_metrics) > 0:
                _check_2d_and_shape1("point_metrics", point_metrics, 1)

            self._point_metrics.append(point_metrics)

    def _process_polygons(self, polygons, coordinates=None):
        """
        Helper to process facets, in case they are given as coordinates form.
        Coordinates will be added to points using add_points() and they will
        be turned into int list that references added points.
        """
        if isinstance(polygons, np.ndarray):
            polygons = polygons.tolist()

        if not isinstance(polygons, tuple | list):
            raise TypeError(
                "facets/polygons should be tuple, list, or np.ndarray"
            )

        if coordinates is None:
            coordinates = isinstance(polygons[0][0], tuple | list | np.ndarray)

        if coordinates:
            point_id_offset = len(self._points)
            id_based_polygons = []
            for coords in polygons:
                # add_points will check for valid input
                self.add_points(coords)

                # point to added points
                id_based_polygons.append(
                    list(range(point_id_offset, point_id_offset + len(coords)))
                )

                point_id_offset += len(coords)

            # at this point, we overwrite polygons
            polygons = id_based_polygons

        return polygons

    def add_facets(
        self, polygons, facet_id=None, facet_constraints=None, coordinates=None
    ):
        """
        Adds facets with one polygon per facet.

        Parameters
        ----------
        polygons: array-like
          list of polygons by coordinate or indices referring to points list.
          coordinates are expected to have [[[x1_1, y1_2, z1_3], ...], ...]
          and indices [[p1_1, p1_2, ...], ... ].
        facet_id: int
          value used to set facetmarker. Default is -123454321.
          can be also interpreted as boundary id as all the output faces on
          this facet will have this id.
        facet_constraints: float or list
          max area for given facets. If it is list,
          len(facet_constraints) == len(polygons) should hold.
        coordinates: bool
          Default is None. Can specify if given polygons are coordinate or
          index based. If None, will check the first entry and determine.

        Returns
        -------
        None
        """
        facet_id_None = False
        if facet_id is None:
            facet_id_None = True
            facet_id = int(self.default_facet_id)

        # make sure polygons are list of list referring to points
        processed_polygon_ids = self._process_polygons(polygons, coordinates)
        n_polygons = len(processed_polygon_ids)

        # by now, we should have points and ids.
        self._facets.extend(processed_polygon_ids)

        # facet_ids
        single_facet_id = True
        if isinstance(facet_id, int | float):
            self._facet_markers.extend([facet_id] * n_polygons)
        elif isinstance(facet_id, tuple | list | np.ndarray):
            single_facet_id = False
            if isinstance(facet_id, np.ndarray):
                facet_id = facet_id.ravel().tolist()

            # assumes flat list
            if len(facet_id) != n_polygons:
                raise ValueError(
                    "For multiple facet_id inputs, "
                    f"len(facet_id)->({len(facet_id)}) should match "
                    f"len(polygons)->({n_polygons})."
                )
            self._facet_markers.extend(facet_id)
        else:
            raise TypeError("facet_id should be int or array-like.")

        # add offset
        self._facet_id_offset += n_polygons

        if facet_constraints is not None:
            # make sure this facet has special marker
            if facet_id_None:
                raise ValueError(
                    "Please specify `facet_id` to apply `facet_constaints`"
                )

            # you can apply multiple constraints - you must have provided
            # multiple facet_ids.
            if isinstance(facet_constraints, tuple | list | np.ndarray):
                if single_facet_id:
                    raise ValueError(
                        "Can't accept multiple facet_constaints, if only "
                        "one facet_id is given."
                    )
                f_constraints = np.asanyarray(facet_constraints).reshape(-1, 2)
                # same length check
                if len(f_constraints) != n_polygons:
                    raise ValueError(
                        "multiple facet constrains should have same len as "
                        f"polygons. polygons-({n_polygons}) / "
                        f"facet_constraints-({len(facet_constraints)})."
                    )

            elif isinstance(facet_constraints, int | float):
                # single facet id -> single constraint
                if single_facet_id:
                    f_constraints = np.array([[facet_id, facet_constraints]])
                else:
                    f_constraints = np.empty((n_polygons, 2), dtype=np.float64)
                    f_constraints[:, 0] = facet_id  # multiple.
                    f_constraints[:, 1] = facet_constraints

            else:
                raise TypeError(
                    "facet_constraints should be array-like or float. "
                    f"Given {type(facet_constraints)}."
                )

            self._facet_constraints.append(f_constraints)

    def add_facet_with_holes(
        self,
        polygons,
        holes,
        facet_id=None,
        coordinates=None,
        facet_constraint=None,
    ):
        """
        Supports full features of tetgenio::facet.

        Parameters
        ----------
        polygons: array-like
          same as add_facets
        holes: array-like
          list of locations to poke holes in this facet. If you don't want
          holes, you can set it to None.
        facet_id: int
        coordinates: bool

        Returns
        -------
        None
        """
        # default facet_id
        facet_id_None = False
        if facet_id is None:
            facet_id_None = True
            facet_id = int(self.default_facet_id)

        # facets
        processed_polygon_ids = self._process_polygons(polygons, coordinates)

        facet_with_holes = [processed_polygon_ids]

        # holes
        if holes is not None:
            holes = np.asanyarray(holes)
            _check_2d_and_shape1("facet holes", holes, 3)
            facet_with_holes.append(holes)
        else:
            # empty hole
            facet_with_holes.append([])

        # facet_id
        facet_with_holes.append(int(facet_id))

        self._facet_with_holes.append(facet_with_holes)

        # optional facet constraint
        if facet_constraint is not None:
            # make sure this facet has special marker
            if facet_id_None:
                raise ValueError(
                    "Please specify `facet_id` to apply `facet_constaints`"
                )

            if not isinstance(facet_constraint, int | float):
                raise TypeError("facet_constraint should be float.")

            self._facet_with_holes_constraints.append(
                [facet_id, facet_constraint]
            )

    def add_holes(self, holes):
        """
        Adds locations to poke holes. This hole propagates to the nearest
        edges of surrounding facets.

        Parameters
        ----------
        holes: (n, 3) array-like
          places to poke holes

        Returns
        -------
        None
        """
        holes = np.asanyarray(holes)
        _check_2d_and_shape1("holes", holes, 3)

        self._holes.append(holes)

    def add_regions(self, regions):
        """
        Mark regions, their attributes, and their maximum volumes.
        Each region has 5 entries - x,y,z coordinate, (int) attribute, (float)
        maximum tetrahedron volume for this region.

        Parameters
        ----------
        regions: (n, 5) np.ndarray
          [[x1, y1, z1, attribute, max_volume], ...]

        Returns
        -------
        None
        """
        regions = np.asanyarray(regions)
        _check_2d_and_shape1("regions", regions, 5)

        self._regions.append(regions)

    def add_segment_constraints(self, segments, constraint):
        """
        Add segment constraints.

        Parameters
        ----------
        segments: array-like
        constraint: float or array-like
          If array-like, should have same len as segments

        Returns
        -------
        None
        """
        segments = np.asanyarray(segments)
        _check_2d_and_shape1("segments", segments, 2)

        if isinstance(constraint, int | float):
            constraint = np.full((len(segments), 1), constraint)

        elif isinstance(constraint, tuple | list | np.ndarray):
            constraint = np.asanyarray(constraint).reshape(-1, 1)

            if len(constraint) != len(segments):
                raise ValueError(
                    f"Constraints ({len(constraint)}) should have same "
                    f"length as segments ({len(segments)})"
                )
        else:
            raise TypeError(
                f"{type(constraint)} is invalid constraint type. "
                f"Supports float or array-like."
            )

        seg_con = np.hstack((segments, constraint))
        self._segment_constraints.append(seg_con)

    def sofarsogood(self):
        """assert for sanity check"""

        assert isinstance(self._points, np.ndarray)
        assert self.points.ndim == 2
        assert self.points.shape[1] == 3

    def to_tetgenio(self, as_dict=False, debug=False):
        """
        Creates tetgenio based on current properties.

        Parameters
        ----------
        as_dict: bool
          return dict instead of TetgenIO object
        debug: bool
          prints debug messages from cpp side.
        """
        # init all required kwargs
        # set required ones or ones that doesn't need further processing
        pytetio = {
            "points": np.vstack(self._points),
            "point_attributes": [],
            "point_metrics": [],
            "facets": self._facets,
            "facet_markers": self._facet_markers,
            "h_facets": self._facet_with_holes,
            "holes": [],
            "regions": [],
            "facet_constraints": [],
            "segment_constraints": [],
            "debug": False,
        }

        if self._point_attributes:
            pytetio["point_attributes"] = np.vstack(self._point_attributes)

        if self._point_metrics:
            pytetio["point_metrics"] = np.vstack(self._point_metrics)

        if self._holes:
            pytetio["holes"] = np.vstack(self._holes)

        if self._regions:
            pytetio["regions"] = np.vstack(self._regions)

        # combine facet and h_facet constraints
        # apply offset now.
        to_stack = []
        if self._facet_with_holes_constraints:
            hf_constraints = np.asanyarray(self._facet_with_holes_constraints)
            to_stack.append(hf_constraints)

        if self._facet_constraints:
            to_stack.extend(self._facet_constraints)

        if to_stack:
            pytetio["facet_constraints"] = np.vstack(to_stack)

        if self._segment_constraints:
            pytetio["segment_constraints"] = np.vstack(
                self._segment_constraints
            )

        if debug:
            pytetio["debug"] = debug

        # must be c contiguous
        for key, value in pytetio.items():
            if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
                pytetio[key] = np.ascontiguousarray(value)

        if as_dict:
            return pytetio

        tetio = core.TetgenIO()
        tetio.setup_plc(**pytetio)

        return tetio

    def show(self):
        pass
