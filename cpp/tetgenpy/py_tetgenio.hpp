#pragma once

#include <algorithm>

#include <tetgen.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tetgenpy/print.hpp>

namespace tetgenpy {

namespace py = pybind11;

template<typename ValueType>
static bool CheckPyArrayShape(const py::array_t<ValueType> arr,
                              const std::vector<int>& shape,
                              const bool throw_ = true) {
  const std::size_t expected_dim = shape.size();
  if (expected_dim != static_cast<std::size_t>(arr.ndim())) {
    if (!throw_)
      return false;
    PrintAndThrowError("Array dim mismatch.",
                       "Expected -",
                       expected_dim,
                       "Given -",
                       arr.ndim());
  }
  const py::ssize_t* arrshape = arr.shape();
  for (std::size_t i{}; i < expected_dim; ++i) {
    const int& shape_i = shape[i];
    if (shape_i < 0) {
      continue;
    } else {
      if (shape_i != arrshape[i]) {
        if (!throw_)
          return false;
        PrintAndThrowError("Array shape mismatch",
                           "in dimension [",
                           i,
                           "].",
                           "Expected -",
                           shape_i,
                           "Given -",
                           arrshape[i]);
      }
    }
  }
  return true;
}

/// std::string to char*
inline char* ToCharPtr(std::string& str_in) { return &str_in.front(); }

class PyTetgenIo : public tetgenio {
protected:
  bool set_called_ = false;

public:
  using Base_ = tetgenio;

  // define constants
  static const int dim_{3};
  static const int n_region_entries_{5};
  static const int n_facet_constraint_entries_{2};
  static const int n_segment_constraint_entries_{3};
  static const int n_triface_corners_{3};
  static const int n_triface_additional_second_order_nodes_{3};
  static const int n_tet_neighbors_{4};
  static const int n_input_tet_corners_{4};
  static const int n_faces_per_tet_{4};
  static const int n_edges_per_tet_{6};
  static const int n_tet_neighbors_per_face_{2}; // -1 entry for outer face
  static const int n_edges_per_face_{3};
  static const int n_edge_corners_{2};
  static const int n_edge_additional_second_order_nodes_{1};
  static const int n_tet_per_edge_{1};
  static const int n_voroedge_corners_{2};
  static const int n_cell_neighbors_per_vorofacet_{2};
  static const int n_point_metrics_{1}; // manual says it's always 1

  // out flag after tetrahedralize
  bool is_output_ = false;

  // default ctor initializes. dtor clears.
  PyTetgenIo() { Base_::initialize(); }
  PyTetgenIo(py::array_t<REAL> points,
             py::list facets,
             py::array_t<REAL> point_attributes,
             py::array_t<REAL> point_metrics,
             py::array_t<int> facet_markers,
             py::list h_facets,
             py::array_t<REAL> holes,
             py::array_t<REAL> regions,
             py::array_t<REAL> facet_constraints,
             py::array_t<REAL> segment_constraints,
             bool debug = false) {
    Base_::initialize();
    SetupPlc(points,
             facets,
             point_attributes,
             point_metrics,
             facet_markers,
             h_facets,
             holes,
             regions,
             facet_constraints,
             segment_constraints,
             debug);
  }

  /// calls clean_memory() if set_called_ flag is set.
  void ClearPreviousSet() {
    if (set_called_) {
      Base_::clean_memory();
      Base_::initialize();
      set_called_ = false;
    }
  }

  void SetupPoints(py::array_t<REAL> points,
                   py::array_t<REAL> point_attributes,
                   py::array_t<REAL> point_metrics,
                   bool debug) {

    ClearPreviousSet();
    set_called_ = true;

    PrintDebug(debug, "Starting PyTetgenIo::SetupPoints");

    /* numberofpoints, pointlist */
    // points shape test - 3D!
    CheckPyArrayShape(points, {-1, dim_});

    Base_::numberofpoints = static_cast<int>(points.shape(0));
    PrintDebug(debug, "set numberofpoints:", Base_::numberofpoints);

    Base_::pointlist = new REAL[Base_::numberofpoints * dim_];
    std::copy_n(static_cast<REAL*>(points.request().ptr),
                Base_::numberofpoints * dim_,
                Base_::pointlist);
    PrintDebug(debug, "set pointlist.");

    /* numberofpointattributes, pointattributelist */
    const int point_attributes_size = static_cast<int>(point_attributes.size());
    if (point_attributes_size > 0) {
      CheckPyArrayShape(point_attributes, {Base_::numberofpoints, -1});

      Base_::numberofpointattributes =
          static_cast<int>(point_attributes.shape(1));
      PrintDebug(debug,
                 "set numberofpointattributes:",
                 Base_::numberofpointattributes);

      Base_::pointattributelist = new REAL[point_attributes_size];
      std::copy_n(static_cast<REAL*>(point_attributes.request().ptr),
                  point_attributes_size,
                  Base_::pointattributelist);
      PrintDebug(debug, "set pointattributelist");
    }

    /* numberofpointmtrs, pointmtrlist */
    const int point_metrics_size = static_cast<int>(point_metrics.size());
    if (point_metrics_size > 0) {
      CheckPyArrayShape(point_metrics,
                        {Base_::numberofpoints, n_point_metrics_});

      Base_::numberofpointmtrs = n_point_metrics_;
      PrintDebug(debug, "set numberofpointmtrs");

      Base_::pointmtrlist = new REAL[point_metrics_size];
      std::copy_n(static_cast<REAL*>(point_metrics.request().ptr),
                  point_metrics_size,
                  Base_::pointmtrlist);
      PrintDebug(debug, "set pointmtrlist");
    }
  }

  /// load full set of input
  /// split input facets into 2 types:
  ///   1) facets without a hole
  ///   2) facets with holes (we will refer to this as h_facet)
  ///
  /// points: (a, 3) np.ndarray
  ///   all points that appear for tetgen query
  /// point_attributes: (a, n) np.ndarray
  ///   attributes for each points. with "-w" switch, first attribute will be
  ///   used as weights for weighted DT
  /// point_metrics: (a, 1) np.ndarray
  ///   sizing function defined at nodes. "-m" switch. Can have zero entries
  ///   for points where you don't want to apply any sizing function.
  /// facets: list
  ///   coplanar single polygons
  /// facet_markers: (len(facets),) np.ndarray
  ///   facet marker. If this facet is at boundary, this will be boundary id.
  ///   len(facet_markers) == len(facets) should hold
  /// h_facets: list
  ///   nested list for self contained facet description with holes list,
  ///   as well as marker.
  ///   [list of h_facet polygons, list of holes, facet marker]
  /// holes: (b, 3) np.ndarray
  ///   list of holes. this pokes holes.
  /// regions: (c, 5) np.ndarray
  ///   list of regions.
  ///   coordinate to mark the region, region attribute, max volume
  void SetupPlc(py::array_t<REAL> points,
                py::list facets,
                py::array_t<REAL> point_attributes,
                py::array_t<REAL> point_metrics,
                py::array_t<int> facet_markers,
                py::list h_facets,
                py::array_t<REAL> holes,
                py::array_t<REAL> regions,
                py::array_t<REAL> facet_constraints,
                py::array_t<REAL> segment_constraints,
                bool debug = false) {

    PrintDebug(debug, "Starting PyTetgenIo::SetupPlc");
    // this call clears tetgenio
    SetupPoints(points, point_attributes, point_metrics, debug);

    // define some constants
    const int n_polygon_per_facet{1};

    /* numberoffacets, facetlist, facetmarkerlist */
    const int n_facets = static_cast<int>(facets.size());
    const int n_facet_markers = static_cast<int>(facet_markers.size());
    const int n_h_facets = static_cast<int>(h_facets.size());
    Base_::numberoffacets = n_facets + n_h_facets;
    if (Base_::numberoffacets > 0) {
      Base_::facetlist = new Base_::facet[Base_::numberoffacets];
    }
    PrintDebug(debug, "set numberoffacets:", Base_::numberoffacets);
    PrintDebug(debug, "-> number of regular facets:", n_facets);
    PrintDebug(debug, "-> number of facets with holes:", n_h_facets);

    // following sizes should match
    // facet markers for h_facets will be added later
    if (n_facet_markers != 0 && n_facets != n_facet_markers) {
      PrintAndThrowError("facets and facet_markers should have same size.",
                         "facets:",
                         n_facets,
                         "/ facet_markers:",
                         n_facet_markers);
    }

    // loop and fillout facets
    // pointers for easy handling of raw ptr elements
    Base_::facet* f;
    Base_::polygon* p;
    for (int i{}; i < n_facets; ++i) {

      // TODO does this work without any issue? else, try
      // reinterpret_steal
      const auto polygon_connec = facets[i].cast<py::list>();

      // get facet and init
      f = &Base_::facetlist[i];
      Base_::init(f);

      // fill polygon
      f->numberofpolygons = n_polygon_per_facet;
      f->polygonlist = new Base_::polygon[n_polygon_per_facet];
      p = &f->polygonlist[0]; // n_polygon_per_facet is 1
      Base_::init(p);
      const int n_polygon_vertices = static_cast<int>(polygon_connec.size());
      p->numberofvertices = n_polygon_vertices;
      p->vertexlist = new int[n_polygon_vertices];

      int j{};
      for (py::handle vertex_id : polygon_connec) {
        p->vertexlist[j] = vertex_id.cast<int>();
        ++j;
      }
    }
    PrintDebug(debug, "set facetlist.");

    // set facet markers. h_facets always carry a marker, so the list is
    // needed as soon as there is any h_facet.
    if (n_facet_markers != 0 || n_h_facets > 0) {
      Base_::facetmarkerlist = new int[Base_::numberoffacets]();
      if (n_facet_markers != 0) {
        std::copy_n(static_cast<int*>(facet_markers.request().ptr),
                    n_facets,
                    Base_::facetmarkerlist);
      }
      PrintDebug(debug, "set facetmarkerlist for facets.");
    }

    // fill h_facets
    int h{n_facets}; // hfacet counter
    for (py::handle hf : h_facets) {
      // index hints
      // 0. -> list of list (polygons; int)
      // 1. -> list of list (holes; double)
      // 2. -> facet marker (;int)
      const auto hf_description = hf.cast<py::list>();
      const auto list_of_polygons = hf_description[0].cast<py::list>();
      const auto list_of_holes = hf_description[1].cast<py::list>();
      const int facet_mark = hf_description[2].cast<int>();

      f = &Base_::facetlist[h];
      Base_::init(f);

      const int n_poly_per_hf = static_cast<int>(list_of_polygons.size());
      f->numberofpolygons = n_poly_per_hf;
      f->polygonlist = new Base_::polygon[n_poly_per_hf];

      // 0
      int i{}; // polygon counter
      for (py::handle poly : list_of_polygons) {
        p = &f->polygonlist[i];
        Base_::init(p);
        const auto poly_vertexlist = poly.cast<py::list>();
        const int n_vertices_per_poly =
            static_cast<int>(poly_vertexlist.size());
        p->numberofvertices = n_vertices_per_poly;
        p->vertexlist = new int[n_vertices_per_poly];

        // fill vertices
        int j{};
        for (py::handle vertex_id : poly_vertexlist) {
          p->vertexlist[j] = vertex_id.cast<int>();
          ++j;
        }
        ++i;
      }

      // 1
      const int n_hf_holes = static_cast<int>(list_of_holes.size());
      if (n_hf_holes > 0) {
        f->numberofholes = n_hf_holes;
        f->holelist = new REAL[n_hf_holes * dim_];
        i = 0;
        for (py::handle hole : list_of_holes) {
          // this needs to have len 3
          int j{};
          const auto hole_coord = hole.cast<py::list>();
          if (hole_coord.size() != 3) {
            PrintAndThrowError("h_facet holes must be 3D coordinates.",
                               "Given hole has the size of (",
                               hole_coord.size(),
                               ")");
          }
          for (py::handle coord : hole_coord) {
            f->holelist[i * dim_ + j] = coord.cast<REAL>();
            ++j;
          }
          ++i;
        }
      }

      // 2
      Base_::facetmarkerlist[h] = facet_mark;

      // next h_facet
      ++h;
    }

    /* numberofholes, holelist */
    const int holes_size = static_cast<int>(holes.size());
    if (holes_size > 0) {
      PrintDebug(debug, "setting  holes.");
      // shape check
      CheckPyArrayShape(holes, {-1, dim_});

      Base_::numberofholes = holes_size / dim_;
      PrintDebug(debug, "->", Base_::numberofholes, "holes.");
      Base_::holelist = new REAL[holes_size];
      std::copy_n(static_cast<REAL*>(holes.request().ptr),
                  holes_size,
                  Base_::holelist);
    }

    /* numberofregions, regionlist */
    const int regions_size = static_cast<int>(regions.size());
    if (regions_size > 0) {
      PrintDebug(debug, "setting regions.");
      // shape check
      CheckPyArrayShape(regions, {-1, n_region_entries_});

      Base_::numberofregions = regions_size / n_region_entries_;
      PrintDebug(debug, "->", Base_::numberofregions, "regions.");
      Base_::regionlist = new REAL[regions_size];
      std::copy_n(static_cast<REAL*>(regions.request().ptr),
                  regions_size,
                  Base_::regionlist);
    }

    /* numberoffacetconstraints, facetconstaintlist */
    const int f_constraints_size = static_cast<int>(facet_constraints.size());
    if (f_constraints_size > 0) {
      PrintDebug(debug, "setting facet constraints.");

      CheckPyArrayShape(facet_constraints, {-1, n_facet_constraint_entries_});
      Base_::numberoffacetconstraints =
          f_constraints_size / n_facet_constraint_entries_;
      PrintDebug(debug,
                 "-> set (",
                 Base_::numberoffacetconstraints,
                 ") numberoffacetconstraints.");

      Base_::facetconstraintlist = new REAL[f_constraints_size];
      std::copy_n(static_cast<REAL*>(facet_constraints.request().ptr),
                  f_constraints_size,
                  Base_::facetconstraintlist);
      PrintDebug(debug, "-> set facetconstraintlist.");
    }

    /* numberofsegmentconstraints, segmentconstraintlist */
    const int s_constraints_size = static_cast<int>(segment_constraints.size());
    if (s_constraints_size > 0) {
      PrintDebug(debug, "setting segment constraints.");

      CheckPyArrayShape(segment_constraints,
                        {-1, n_segment_constraint_entries_});
      Base_::numberofsegmentconstraints =
          s_constraints_size / n_segment_constraint_entries_;
      PrintDebug(debug,
                 "-> set (",
                 Base_::numberofsegmentconstraints,
                 ") numberofsegmentconstraints.");

      Base_::segmentconstraintlist = new REAL[s_constraints_size];
      std::copy_n(static_cast<REAL*>(segment_constraints.request().ptr),
                  s_constraints_size,
                  Base_::segmentconstraintlist);
      PrintDebug(debug, "-> set segmentconstraintlist");
    }
  }

  /// Setup Existing TetMesh, perhaps to refine, perhaps to coarsen,
  /// perhaps as a background mesh.
  void SetupTetMesh(py::array_t<REAL> points,
                    py::array_t<int> tetrahedra, // (n, 4)
                    py::array_t<REAL> point_attributes,
                    py::array_t<REAL> point_metrics,
                    py::array_t<REAL> tetrahedron_attributes,
                    py::array_t<REAL> tetrahedron_constraints, // tetvolumelist
                    py::array_t<int> refine_elements,          // (n, 4)
                    py::array_t<REAL> refine_element_constraints, // (n, 1)
                    py::array_t<int> trifaces,
                    py::array_t<int> triface_markers,
                    py::array_t<int> edges,
                    py::array_t<int> edge_markers,
                    bool debug) {
    PrintDebug(debug, "Starting PyTetgenIo::SetupTetMesh");
    // this call clears tetgenio
    SetupPoints(points, point_attributes, point_metrics, debug);

    /* tetrahedronlist, numberoftetrahdra */
    // for input, only take linear tets
    CheckPyArrayShape(tetrahedra, {-1, n_input_tet_corners_});

    Base_::numberoftetrahedra = static_cast<int>(tetrahedra.shape(0));
    PrintDebug(debug, "set numberoftetrahedra:", Base_::numberoftetrahedra);

    const int tetlist_size = Base_::numberoftetrahedra * n_input_tet_corners_;
    Base_::tetrahedronlist = new int[tetlist_size];
    std::copy_n(static_cast<int*>(tetrahedra.request().ptr),
                tetlist_size,
                Base_::tetrahedronlist);
    PrintDebug(debug, "set tetrahedronlist.");

    /* tetrahedronattributelist, numberoftetrahedronattributes */
    const int tet_attr_size = static_cast<int>(tetrahedron_attributes.size());
    if (tet_attr_size > 0) {
      // just needs to be 2D
      CheckPyArrayShape(tetrahedron_attributes,
                        {Base_::numberoftetrahedra, -1});

      Base_::numberoftetrahedronattributes =
          static_cast<int>(tetrahedron_attributes.shape(1));
      PrintDebug(debug,
                 "set numberoftetrahedronattributes:",
                 Base_::numberoftetrahedronattributes);

      Base_::tetrahedronattributelist = new REAL[tet_attr_size];
      std::copy_n(static_cast<REAL*>(tetrahedron_attributes.request().ptr),
                  tet_attr_size,
                  Base_::tetrahedronattributelist);
      PrintDebug(debug, "set tetrahedronattributelist.");
    }

    /* tetrahedronvolumelist */
    const int tet_vol_size = static_cast<int>(tetrahedron_constraints.size());
    if (tet_vol_size > 0) {
      CheckPyArrayShape(tetrahedron_constraints,
                        {Base_::numberoftetrahedra, 1});

      Base_::tetrahedronvolumelist = new REAL[tet_vol_size];
      std::copy_n(static_cast<REAL*>(tetrahedron_constraints.request().ptr),
                  tet_vol_size,
                  Base_::tetrahedronvolumelist);
      PrintDebug(debug, "set tetrahedronvolumelist.");
    }

    /* refine_elem_list, refine_elem_vol_list, numberofrefineelems */
    const int refine_elem_size = static_cast<int>(refine_elements.size());
    if (refine_elem_size > 0) {
      // size check
      CheckPyArrayShape(refine_elements, {-1, n_input_tet_corners_});
      Base_::numberofrefineelems = static_cast<int>(refine_elements.shape(0));
      PrintDebug(debug, "set numberofrefineelems:", Base_::numberofrefineelems);

      CheckPyArrayShape(refine_element_constraints,
                        {Base_::numberofrefineelems, 1});
      Base_::refine_elem_list = new int[refine_elem_size];
      std::copy_n(static_cast<int*>(refine_elements.request().ptr),
                  refine_elem_size,
                  Base_::refine_elem_list);
      PrintDebug(debug, "set refine_elem_list.");

      Base_::refine_elem_vol_list = new REAL[Base_::numberofrefineelems];
      std::copy_n(static_cast<REAL*>(refine_element_constraints.request().ptr),
                  Base_::numberofrefineelems,
                  Base_::refine_elem_vol_list);
      PrintDebug(debug, "set refine_elem_vol_list.");
    }

    /* trifacelist, trifacemarkerlist, numberoftrifaces */
    const int trifaces_size = static_cast<int>(trifaces.size());
    if (trifaces_size > 0) {
      CheckPyArrayShape(trifaces, {-1, n_triface_corners_});
      Base_::numberoftrifaces = static_cast<int>(trifaces.shape(0));
      PrintDebug(debug, "set numberoftrifaces:", Base_::numberoftrifaces);
      Base_::trifacelist = new int[trifaces_size];
      std::copy_n(static_cast<int*>(trifaces.request().ptr),
                  trifaces_size,
                  Base_::trifacelist);
      PrintDebug(debug, "set trifacelist.");

      const int triface_markers_size = static_cast<int>(triface_markers.size());
      if (triface_markers_size > 0) {
        CheckPyArrayShape(triface_markers, {Base_::numberoftrifaces, 1});
        Base_::trifacemarkerlist = new int[triface_markers_size];
        std::copy_n(static_cast<int*>(triface_markers.request().ptr),
                    triface_markers_size,
                    Base_::trifacemarkerlist);
        PrintDebug(debug, "set trifacemarkerlist.");
      }
    }

    /* edgelist, edgemarkerlist, numberofedges */
    const int edges_size = static_cast<int>(edges.size());
    if (edges_size > 0) {
      CheckPyArrayShape(edges, {-1, n_edge_corners_});
      Base_::numberofedges = static_cast<int>(edges.shape(0));
      PrintDebug(debug, "set numberofedges:", Base_::numberofedges);

      Base_::edgelist = new int[edges_size];
      std::copy_n(static_cast<int*>(edges.request().ptr),
                  edges_size,
                  Base_::edgelist);
      PrintDebug(debug, "set edgelist.");

      const int edge_markers_size = static_cast<int>(edge_markers.size());
      if (edge_markers_size > 0) {
        CheckPyArrayShape(edge_markers, {Base_::numberofedges, 1});

        Base_::edgemarkerlist = new int[edge_markers_size];
        std::copy_n(static_cast<int*>(edge_markers.request().ptr),
                    edge_markers_size,
                    Base_::edgemarkerlist);
        PrintDebug(debug, "set edgemarkerlist");
      }
    }
  }

  /* getters */
  template<typename DataType>
  py::array_t<DataType> CopyFromBase(const int base_data_count,
                                     const int base_data_stride,
                                     const DataType* base_array_ptr) {

    // return if zero
    if (base_data_count == 0 || base_array_ptr == (DataType*) NULL) {
      return py::array_t<DataType>(0);
    }

    const int base_array_len = base_data_count * base_data_stride;
    py::array_t<DataType> output_array({base_data_count, base_data_stride});
    std::copy_n(base_array_ptr,
                base_array_len,
                static_cast<DataType*>(output_array.request().ptr));

    return output_array;
  }

  // I / O
  py::array_t<REAL> GetPoints() {
    return CopyFromBase(Base_::numberofpoints, dim_, Base_::pointlist);
  }

  // I / O
  py::array_t<REAL> GetPointAttributes() {
    return CopyFromBase(Base_::numberofpoints,
                        Base_::numberofpointattributes,
                        Base_::pointattributelist);
  }

  // I - only for refinement / O
  py::array_t<int> GetTetrahedra() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        Base_::numberofcorners,
                        Base_::tetrahedronlist);
  }

  // I - only for refinement / O
  py::array_t<REAL> GetTetrahedronAttributes() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        Base_::numberoftetrahedronattributes,
                        Base_::tetrahedronattributelist);
  }

  // I - for refinement / O
  py::array_t<int> GetTriFaces() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_triface_corners_,
                        Base_::trifacelist);
  }

  py::array_t<int> GetTriFaceMarkers() {
    return CopyFromBase(Base_::numberoftrifaces, 1, Base_::trifacemarkerlist);
  }

  py::array_t<int> GetO2Faces() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_triface_additional_second_order_nodes_,
                        Base_::o2facelist);
  }

  py::array_t<int> GetNeighbors() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_tet_neighbors_,
                        Base_::neighborlist);
  }

  py::array_t<int> GetTet2Faces() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_faces_per_tet_,
                        Base_::tet2facelist);
  }

  py::array_t<int> GetTet2Edges() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_edges_per_tet_,
                        Base_::tet2edgelist);
  }

  py::array_t<int> GetFace2Tets() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_tet_neighbors_per_face_,
                        Base_::face2tetlist);
  }

  py::array_t<int> GetFace2Edges() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_edges_per_face_,
                        Base_::face2edgelist);
  }

  py::array_t<int> GetEdges() {
    return CopyFromBase(Base_::numberofedges, n_edge_corners_, Base_::edgelist);
  }

  py::array_t<int> GetEdgeMarkers() {
    return CopyFromBase(Base_::numberofedges, 1, Base_::edgemarkerlist);
  }

  py::array_t<int> GetO2Edges() {
    return CopyFromBase(Base_::numberofedges,
                        n_edge_additional_second_order_nodes_,
                        Base_::o2edgelist);
  }

  py::array_t<int> GetEdge2Tets() {
    return CopyFromBase(Base_::numberofedges,
                        n_tet_per_edge_,
                        Base_::edge2tetlist);
  }

  // Voronoi point, edges, facets, cells.
  // currently tetgen.cxx does not parse "v" switch, thus only available with
  // tetgenbehavior
  py::dict GetVoronoi() {
    //
    py::dict voronoi;

    // points are simple copy
    voronoi["points"] =
        CopyFromBase(Base_::numberofvpoints, dim_, Base_::vpointlist);

    // vornoi edges
    py::array_t<int> vedges({Base_::numberofvedges, n_voroedge_corners_});
    int* vedges_ptr = static_cast<int*>(vedges.request().ptr);

    // infinite vertex is not defined.
    // however for plotting, one could project this to a certain plane.
    // to make that process a bit easier, following are provided:
    // 1. reference to infinite vertex is assigned with negative value
    //    and it counts down at every occurrence. that way, we can just
    //    append new points to existing point array.
    // 2. keep track of edge ids with infinite vertex
    // 3. and their normals
    int infinite_vertex_id{-1}; //
    py::list edges_with_infinite_vertex;
    py::list ray_directions;
    for (int i{}; i < Base_::numberofvedges; ++i) {
      const auto& base_voroedge = Base_::vedgelist[i];

      const int ve_id = i * n_voroedge_corners_;
      vedges_ptr[ve_id] = base_voroedge.v1;

      const auto& v2 = base_voroedge.v2;
      if (v2 < 0) { // negative -> "infinite vertex"
        vedges_ptr[ve_id + 1] = infinite_vertex_id; // <- (1)
        --infinite_vertex_id;

        edges_with_infinite_vertex.append(i); // <- (2)

        py::list normal(3);
        normal[0] = base_voroedge.vnormal[0];
        normal[1] = base_voroedge.vnormal[1];
        normal[2] = base_voroedge.vnormal[2];
        ray_directions.append(normal); // <- 3
      } else {
        vedges_ptr[ve_id + 1] = v2;
      }
    }
    voronoi["edges"] = vedges;
    voronoi["edges_with_infinite_vertex"] =
        py::make_tuple(edges_with_infinite_vertex, ray_directions);

    // facets
    // Base_::vorofacet includes cell neighbors too
    py::array_t<int> facet2cell(
        {Base_::numberofvfacets, n_cell_neighbors_per_vorofacet_});
    py::list facets;
    int* f2c_ptr = static_cast<int*>(facet2cell.request().ptr);
    for (int i{}; i < Base_::numberofvfacets; ++i) {
      const auto& v_facet = Base_::vfacetlist[i];
      const int f2c_id = i * n_cell_neighbors_per_vorofacet_;
      f2c_ptr[f2c_id] = v_facet.c1;
      f2c_ptr[f2c_id + 1] = v_facet.c2;

      py::list facet;
      // "... elist[0] saves the number of Voronoi edges ..."
      for (int j{1}; j < v_facet.elist[0] + 1; ++j) {
        facet.append(v_facet.elist[j]);
      }
      facets.append(facet);
    }
    voronoi["facet2cell"] = facet2cell;
    voronoi["facets"] = facets;

    // cells
    py::list cells;
    for (int i{}; i < Base_::numberofvcells; ++i) {
      const auto& v_cell = Base_::vcelllist[i];
      const auto& n_facets = v_cell[0];
      py::list cell;
      for (int j{1}; j < v_cell[0] + 1; ++j) {
        cell.append(v_cell[j]);
      }
      cells.append(cell);
    }
    voronoi["cells"] = cells;

    return voronoi;
  }

  // load from files
  void LoadNode(std::string fbase) { Base_::load_node(ToCharPtr(fbase)); }

  void LoadEdge(std::string fbase) { Base_::load_edge(ToCharPtr(fbase)); }

  void LoadFace(std::string fbase) { Base_::load_face(ToCharPtr(fbase)); }

  void LoadTet(std::string fbase) { Base_::load_tet(ToCharPtr(fbase)); }

  void LoadVol(std::string fbase) { Base_::load_vol(ToCharPtr(fbase)); }

  void LoadVar(std::string fbase) { Base_::load_var(ToCharPtr(fbase)); }

  void LoadMtr(std::string fbase) { Base_::load_mtr(ToCharPtr(fbase)); }

  void LoadElem(std::string fbase) { Base_::load_elem(ToCharPtr(fbase)); }

  void LoadPoly(std::string fbase) { Base_::load_poly(ToCharPtr(fbase)); }

  void LoadOff(std::string fbase) { Base_::load_off(ToCharPtr(fbase)); }

  void LoadPly(std::string fbase) { Base_::load_ply(ToCharPtr(fbase)); }

  void LoadStl(std::string fbase) { Base_::load_stl(ToCharPtr(fbase)); }

  void LoadVtk(std::string fbase) { Base_::load_vtk(ToCharPtr(fbase)); }

  void LoadMedit(std::string fbase, bool istetmesh) {
    Base_::load_medit(ToCharPtr(fbase), (istetmesh) ? 1 : 0);
  }

  // save to files
  void SaveNodes(std::string fbase) { Base_::save_nodes(ToCharPtr(fbase)); }

  void SaveElements(std::string fbase) {
    Base_::save_elements(ToCharPtr(fbase));
  }

  void SaveFaces(std::string fbase) { Base_::save_faces(ToCharPtr(fbase)); }

  void SaveEdges(std::string fbase) { Base_::save_edges(ToCharPtr(fbase)); }

  void SaveNeighbors(std::string fbase) {
    Base_::save_neighbors(ToCharPtr(fbase));
  }

  void SavePoly(std::string fbase) { Base_::save_poly(ToCharPtr(fbase)); }

  void SaveFaces2Smesh(std::string fbase) {
    Base_::save_faces2smesh(ToCharPtr(fbase));
  }
};

inline void add_pytetgenio_class(py::module_& m) {

  py::class_<PyTetgenIo> klasse(m, "TetgenIO");

  klasse.def(py::init<>())
      .def("setup_points",
           &PyTetgenIo::SetupPoints,
           py::arg("points"),
           py::arg("point_attributes") = py::array_t<REAL>(),
           py::arg("point_metrics") = py::array_t<REAL>(),
           py::arg("debug") = false)
      .def("setup_plc",
           &PyTetgenIo::SetupPlc,
           py::arg("points"),
           py::arg("facets"),
           py::arg("point_attributes") = py::array_t<REAL>(),
           py::arg("point_metrics") = py::array_t<REAL>(),
           py::arg("facet_markers") = py::array_t<int>(),
           py::arg("h_facets") = py::list(),
           py::arg("holes") = py::array_t<REAL>(),
           py::arg("regions") = py::array_t<REAL>(),
           py::arg("facet_constraints") = py::array_t<REAL>(),
           py::arg("segment_constraints") = py::array_t<REAL>(),
           py::arg("debug") = false)
      .def("setup_tetmesh",
           &PyTetgenIo::SetupTetMesh,
           py::arg("points"),
           py::arg("tetrahedra"),
           py::arg("point_attributes") = py::array_t<REAL>(),
           py::arg("point_metrics") = py::array_t<REAL>(),
           py::arg("tetrahedron_attributes") = py::array_t<REAL>(),
           py::arg("tetrahedron_constraints") = py::array_t<REAL>(),
           py::arg("refine_elements") = py::array_t<int>(),
           py::arg("refine_element_constraints") = py::array_t<REAL>(),
           py::arg("trifaces") = py::array_t<int>(),
           py::arg("triface_markers") = py::array_t<int>(),
           py::arg("edges") = py::array_t<int>(),
           py::arg("edge_markers") = py::array_t<int>(),
           py::arg("debug") = false)
      .def("points", &PyTetgenIo::GetPoints)
      .def("pointattributes", &PyTetgenIo::GetPointAttributes)
      .def("tetrahedra", &PyTetgenIo::GetTetrahedra)
      .def("tetrahedronattributes", &PyTetgenIo::GetTetrahedronAttributes)
      .def("trifaces", &PyTetgenIo::GetTriFaces)
      .def("trifacemarkers", &PyTetgenIo::GetTriFaceMarkers)
      .def("neighbors", &PyTetgenIo::GetNeighbors)
      .def("tet2faces", &PyTetgenIo::GetTet2Faces)
      .def("tet2edges", &PyTetgenIo::GetTet2Edges)
      .def("face2tets", &PyTetgenIo::GetFace2Tets)
      .def("face2edges", &PyTetgenIo::GetFace2Edges)
      .def("edges", &PyTetgenIo::GetEdges)
      .def("edgemarkers", &PyTetgenIo::GetEdgeMarkers)
      .def("edge2tets", &PyTetgenIo::GetEdge2Tets)
      .def("voronoi", &PyTetgenIo::GetVoronoi)
      .def("load_node", &PyTetgenIo::LoadNode, py::arg("fname_base"))
      .def("load_edge", &PyTetgenIo::LoadEdge, py::arg("fname_base"))
      .def("load_face", &PyTetgenIo::LoadFace, py::arg("fname_base"))
      .def("load_tet", &PyTetgenIo::LoadTet, py::arg("fname_base"))
      .def("load_vol", &PyTetgenIo::LoadVol, py::arg("fname_base"))
      .def("load_var", &PyTetgenIo::LoadVar, py::arg("fname_base"))
      .def("load_mtr", &PyTetgenIo::LoadMtr, py::arg("fname_base"))
      .def("load_elem", &PyTetgenIo::LoadElem, py::arg("fname_base"))
      .def("load_poly", &PyTetgenIo::LoadPoly, py::arg("fname_base"))
      .def("load_off", &PyTetgenIo::LoadOff, py::arg("fname_base"))
      .def("load_ply", &PyTetgenIo::LoadPly, py::arg("fname_base"))
      .def("load_stl", &PyTetgenIo::LoadStl, py::arg("fname_base"))
      .def("load_vtk", &PyTetgenIo::LoadVtk, py::arg("fname_base"))
      .def("load_medit",
           &PyTetgenIo::LoadMedit,
           py::arg("fname_base"),
           py::arg("is_tet_mesh"))
      .def("save_nodes", &PyTetgenIo::SaveNodes, py::arg("fname_base"))
      .def("save_elements", &PyTetgenIo::SaveElements, py::arg("fname_base"))
      .def("save_faces", &PyTetgenIo::SaveFaces, py::arg("fname_base"))
      .def("save_edges", &PyTetgenIo::SaveEdges, py::arg("fname_base"))
      .def("save_neighbors", &PyTetgenIo::SaveNeighbors, py::arg("fname_base"))
      .def("save_poly", &PyTetgenIo::SavePoly, py::arg("fname_base"))
      .def("save_faces2Smesh",
           &PyTetgenIo::SaveFaces2Smesh,
           py::arg("fname_base"));
}

} // namespace tetgenpy
