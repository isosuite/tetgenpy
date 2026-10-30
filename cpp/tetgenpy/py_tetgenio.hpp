#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <tetgen.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tetgenpy/print.hpp>

namespace tetgenpy {

namespace nb = nanobind;

/// std::string to null-terminated char vector
inline std::vector<char> ToCStr(const std::string& str) {
  std::vector<char> c(str.begin(), str.end());
  c.push_back('\0');
  return c;
}

template<typename T>
class ArrayInput {
public:
  const T* data = nullptr;
  std::vector<int64_t> shape;
  size_t total_size = 0;
  nb::object keep_alive;

  bool empty() const { return total_size == 0; }

  static ArrayInput from_object(nb::handle obj, const char* name = "array") {
    ArrayInput result;
    if (obj.is_none()) {
      return result;
    }

    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
      if (nb::len(obj) == 0) {
        return result;
      }
      nb::object np = nb::module_::import_("numpy");
      nb::object dtype_obj;
      if constexpr (std::is_same_v<T, int>) {
        dtype_obj = np.attr("int32");
      } else if constexpr (std::is_same_v<T, double>) {
        dtype_obj = np.attr("float64");
      } else if constexpr (std::is_same_v<T, float>) {
        dtype_obj = np.attr("float32");
      }
      nb::object arr = np.attr("ascontiguousarray")(obj, dtype_obj);
      return from_ndarray(arr, name);
    }

    return from_ndarray(obj, name);
  }

  static ArrayInput from_ndarray(nb::handle obj, const char* name = "array") {
    ArrayInput result;
    if (obj.is_none()) {
      return result;
    }

    try {
      nb::ndarray<const T, nb::c_contig, nb::device::cpu> arr =
          nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(obj);
      result.total_size = arr.size();
      if (result.total_size == 0) {
        return result;
      }
      result.data = arr.data();
      result.shape.resize(arr.ndim());
      for (size_t i = 0; i < arr.ndim(); ++i) {
        result.shape[i] = static_cast<int64_t>(arr.shape(i));
      }
      return result;
    } catch (...) {
      nb::object np = nb::module_::import_("numpy");
      nb::object dtype_obj;
      if constexpr (std::is_same_v<T, int>) {
        dtype_obj = np.attr("int32");
      } else if constexpr (std::is_same_v<T, double>) {
        dtype_obj = np.attr("float64");
      } else if constexpr (std::is_same_v<T, float>) {
        dtype_obj = np.attr("float32");
      }
      nb::object arr = np.attr("ascontiguousarray")(obj, dtype_obj);
      nb::ndarray<const T, nb::c_contig, nb::device::cpu> casted =
          nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(arr);
      result.keep_alive = arr;
      result.total_size = casted.size();
      if (result.total_size == 0) {
        return result;
      }
      result.data = casted.data();
      result.shape.resize(casted.ndim());
      for (size_t i = 0; i < casted.ndim(); ++i) {
        result.shape[i] = static_cast<int64_t>(casted.shape(i));
      }
      return result;
    }
  }

  void check_shape(const std::vector<int>& expected_shape,
                   const char* name = "Array") const {
    const size_t expected_dim = expected_shape.size();
    if (expected_dim != shape.size()) {
      PrintAndThrowError(name,
                         " dim mismatch. Expected -",
                         expected_dim,
                         " Given -",
                         shape.size());
    }
    for (size_t i = 0; i < expected_dim; ++i) {
      if (expected_shape[i] >= 0 && shape[i] != expected_shape[i]) {
        PrintAndThrowError(name,
                           " shape mismatch in dimension [",
                           i,
                           "]. Expected -",
                           expected_shape[i],
                           " Given -",
                           shape[i]);
      }
    }
  }
};

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
  PyTetgenIo(nb::handle points,
             nb::list facets,
             nb::handle point_attributes = nb::none(),
             nb::handle point_metrics = nb::none(),
             nb::handle facet_markers = nb::none(),
             nb::list h_facets = nb::list(),
             nb::handle holes = nb::none(),
             nb::handle regions = nb::none(),
             nb::handle facet_constraints = nb::none(),
             nb::handle segment_constraints = nb::none(),
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

  void SetupPoints(nb::handle points,
                   nb::handle point_attributes = nb::none(),
                   nb::handle point_metrics = nb::none(),
                   bool debug = false) {

    ClearPreviousSet();
    set_called_ = true;

    PrintDebug(debug, "Starting PyTetgenIo::SetupPoints");

    auto pts = ArrayInput<REAL>::from_object(points, "points");
    pts.check_shape({-1, dim_}, "points");

    Base_::numberofpoints = static_cast<int>(pts.shape[0]);
    PrintDebug(debug, "set numberofpoints:", Base_::numberofpoints);

    Base_::pointlist = new REAL[Base_::numberofpoints * dim_];
    std::copy_n(pts.data, Base_::numberofpoints * dim_, Base_::pointlist);
    PrintDebug(debug, "set pointlist.");

    auto p_attrs =
        ArrayInput<REAL>::from_object(point_attributes, "point_attributes");
    if (!p_attrs.empty()) {
      p_attrs.check_shape({Base_::numberofpoints, -1}, "point_attributes");

      Base_::numberofpointattributes = static_cast<int>(p_attrs.shape[1]);
      PrintDebug(debug,
                 "set numberofpointattributes:",
                 Base_::numberofpointattributes);

      Base_::pointattributelist = new REAL[p_attrs.total_size];
      std::copy_n(p_attrs.data, p_attrs.total_size, Base_::pointattributelist);
      PrintDebug(debug, "set pointattributelist");
    }

    auto p_mtrs = ArrayInput<REAL>::from_object(point_metrics, "point_metrics");
    if (!p_mtrs.empty()) {
      p_mtrs.check_shape({Base_::numberofpoints, n_point_metrics_},
                         "point_metrics");

      Base_::numberofpointmtrs = n_point_metrics_;
      PrintDebug(debug, "set numberofpointmtrs");

      Base_::pointmtrlist = new REAL[p_mtrs.total_size];
      std::copy_n(p_mtrs.data, p_mtrs.total_size, Base_::pointmtrlist);
      PrintDebug(debug, "set pointmtrlist");
    }
  }

  void SetupPlc(nb::handle points,
                nb::list facets,
                nb::handle point_attributes = nb::none(),
                nb::handle point_metrics = nb::none(),
                nb::handle facet_markers = nb::none(),
                nb::list h_facets = nb::list(),
                nb::handle holes = nb::none(),
                nb::handle regions = nb::none(),
                nb::handle facet_constraints = nb::none(),
                nb::handle segment_constraints = nb::none(),
                bool debug = false) {

    PrintDebug(debug, "Starting PyTetgenIo::SetupPlc");
    SetupPoints(points, point_attributes, point_metrics, debug);

    const int n_polygon_per_facet{1};

    const int n_facets = static_cast<int>(facets.size());
    auto f_markers =
        ArrayInput<int>::from_object(facet_markers, "facet_markers");
    const int n_facet_markers =
        f_markers.empty() ? 0 : static_cast<int>(f_markers.total_size);
    const int n_h_facets = static_cast<int>(h_facets.size());
    Base_::numberoffacets = n_facets + n_h_facets;
    if (Base_::numberoffacets > 0) {
      Base_::facetlist = new Base_::facet[Base_::numberoffacets];
    }
    PrintDebug(debug, "set numberoffacets:", Base_::numberoffacets);
    PrintDebug(debug, "-> number of regular facets:", n_facets);
    PrintDebug(debug, "-> number of facets with holes:", n_h_facets);

    if (n_facet_markers != 0 && n_facets != n_facet_markers) {
      PrintAndThrowError("facets and facet_markers should have same size.",
                         "facets:",
                         n_facets,
                         "/ facet_markers:",
                         n_facet_markers);
    }

    Base_::facet* f;
    Base_::polygon* p;
    for (int i{}; i < n_facets; ++i) {
      nb::sequence polygon_connec = nb::cast<nb::sequence>(facets[i]);

      f = &Base_::facetlist[i];
      Base_::init(f);

      f->numberofpolygons = n_polygon_per_facet;
      f->polygonlist = new Base_::polygon[n_polygon_per_facet];
      p = &f->polygonlist[0];
      Base_::init(p);
      const int n_polygon_vertices =
          static_cast<int>(nb::len(polygon_connec));
      p->numberofvertices = n_polygon_vertices;
      p->vertexlist = new int[n_polygon_vertices];

      int j{};
      for (nb::handle vertex_id : polygon_connec) {
        p->vertexlist[j] = nb::cast<int>(vertex_id);
        ++j;
      }
    }
    PrintDebug(debug, "set facetlist.");

    if (n_facet_markers != 0 || n_h_facets > 0) {
      Base_::facetmarkerlist = new int[Base_::numberoffacets]();
      if (n_facet_markers != 0) {
        std::copy_n(f_markers.data, n_facets, Base_::facetmarkerlist);
      }
      PrintDebug(debug, "set facetmarkerlist for facets.");
    }

    int h{n_facets};
    for (nb::handle hf : h_facets) {
      nb::sequence hf_description = nb::cast<nb::sequence>(hf);
      nb::sequence list_of_polygons =
          nb::cast<nb::sequence>(hf_description[0]);
      nb::handle holes_handle = hf_description[1];
      int facet_mark = nb::cast<int>(hf_description[2]);

      f = &Base_::facetlist[h];
      Base_::init(f);

      const int n_poly_per_hf =
          static_cast<int>(nb::len(list_of_polygons));
      f->numberofpolygons = n_poly_per_hf;
      f->polygonlist = new Base_::polygon[n_poly_per_hf];

      int i{};
      for (nb::handle poly : list_of_polygons) {
        p = &f->polygonlist[i];
        Base_::init(p);
        nb::sequence poly_vertexlist = nb::cast<nb::sequence>(poly);
        const int n_vertices_per_poly =
            static_cast<int>(nb::len(poly_vertexlist));
        p->numberofvertices = n_vertices_per_poly;
        p->vertexlist = new int[n_vertices_per_poly];

        int j{};
        for (nb::handle vertex_id : poly_vertexlist) {
          p->vertexlist[j] = nb::cast<int>(vertex_id);
          ++j;
        }
        ++i;
      }

      auto hf_holes_buf =
          ArrayInput<REAL>::from_object(holes_handle, "h_facet holes");
      if (!hf_holes_buf.empty()) {
        hf_holes_buf.check_shape({-1, dim_}, "h_facet holes");
        const int n_hf_holes = static_cast<int>(hf_holes_buf.shape[0]);
        f->numberofholes = n_hf_holes;
        f->holelist = new REAL[n_hf_holes * dim_];
        std::copy_n(hf_holes_buf.data, n_hf_holes * dim_, f->holelist);
      }

      Base_::facetmarkerlist[h] = facet_mark;
      ++h;
    }

    auto holes_buf = ArrayInput<REAL>::from_object(holes, "holes");
    if (!holes_buf.empty()) {
      PrintDebug(debug, "setting holes.");
      holes_buf.check_shape({-1, dim_}, "holes");

      Base_::numberofholes = static_cast<int>(holes_buf.shape[0]);
      PrintDebug(debug, "->", Base_::numberofholes, "holes.");
      Base_::holelist = new REAL[holes_buf.total_size];
      std::copy_n(holes_buf.data, holes_buf.total_size, Base_::holelist);
    }

    auto regions_buf = ArrayInput<REAL>::from_object(regions, "regions");
    if (!regions_buf.empty()) {
      PrintDebug(debug, "setting regions.");
      regions_buf.check_shape({-1, n_region_entries_}, "regions");

      Base_::numberofregions = static_cast<int>(regions_buf.shape[0]);
      PrintDebug(debug, "->", Base_::numberofregions, "regions.");
      Base_::regionlist = new REAL[regions_buf.total_size];
      std::copy_n(regions_buf.data, regions_buf.total_size, Base_::regionlist);
    }

    auto f_constraints_buf =
        ArrayInput<REAL>::from_object(facet_constraints, "facet_constraints");
    if (!f_constraints_buf.empty()) {
      PrintDebug(debug, "setting facet constraints.");
      f_constraints_buf.check_shape({-1, n_facet_constraint_entries_},
                                    "facet_constraints");
      Base_::numberoffacetconstraints =
          static_cast<int>(f_constraints_buf.shape[0]);
      PrintDebug(debug,
                 "-> set (",
                 Base_::numberoffacetconstraints,
                 ") numberoffacetconstraints.");

      Base_::facetconstraintlist = new REAL[f_constraints_buf.total_size];
      std::copy_n(f_constraints_buf.data,
                  f_constraints_buf.total_size,
                  Base_::facetconstraintlist);
      PrintDebug(debug, "-> set facetconstraintlist.");
    }

    auto s_constraints_buf = ArrayInput<REAL>::from_object(
        segment_constraints, "segment_constraints");
    if (!s_constraints_buf.empty()) {
      PrintDebug(debug, "setting segment constraints.");
      s_constraints_buf.check_shape({-1, n_segment_constraint_entries_},
                                    "segment_constraints");
      Base_::numberofsegmentconstraints =
          static_cast<int>(s_constraints_buf.shape[0]);
      PrintDebug(debug,
                 "-> set (",
                 Base_::numberofsegmentconstraints,
                 ") numberofsegmentconstraints.");

      Base_::segmentconstraintlist = new REAL[s_constraints_buf.total_size];
      std::copy_n(s_constraints_buf.data,
                  s_constraints_buf.total_size,
                  Base_::segmentconstraintlist);
      PrintDebug(debug, "-> set segmentconstraintlist");
    }
  }

  void SetupTetMesh(nb::handle points,
                    nb::handle tetrahedra,
                    nb::handle point_attributes = nb::none(),
                    nb::handle point_metrics = nb::none(),
                    nb::handle tetrahedron_attributes = nb::none(),
                    nb::handle tetrahedron_constraints = nb::none(),
                    nb::handle refine_elements = nb::none(),
                    nb::handle refine_element_constraints = nb::none(),
                    nb::handle trifaces = nb::none(),
                    nb::handle triface_markers = nb::none(),
                    nb::handle edges = nb::none(),
                    nb::handle edge_markers = nb::none(),
                    bool debug = false) {
    PrintDebug(debug, "Starting PyTetgenIo::SetupTetMesh");
    SetupPoints(points, point_attributes, point_metrics, debug);

    auto tets_buf = ArrayInput<int>::from_object(tetrahedra, "tetrahedra");
    tets_buf.check_shape({-1, n_input_tet_corners_}, "tetrahedra");

    Base_::numberoftetrahedra = static_cast<int>(tets_buf.shape[0]);
    PrintDebug(debug, "set numberoftetrahedra:", Base_::numberoftetrahedra);

    Base_::tetrahedronlist = new int[tets_buf.total_size];
    std::copy_n(tets_buf.data, tets_buf.total_size, Base_::tetrahedronlist);
    PrintDebug(debug, "set tetrahedronlist.");

    auto tet_attrs = ArrayInput<REAL>::from_object(tetrahedron_attributes,
                                                   "tetrahedron_attributes");
    if (!tet_attrs.empty()) {
      tet_attrs.check_shape({Base_::numberoftetrahedra, -1},
                            "tetrahedron_attributes");

      Base_::numberoftetrahedronattributes =
          static_cast<int>(tet_attrs.shape[1]);
      PrintDebug(debug,
                 "set numberoftetrahedronattributes:",
                 Base_::numberoftetrahedronattributes);

      Base_::tetrahedronattributelist = new REAL[tet_attrs.total_size];
      std::copy_n(tet_attrs.data,
                  tet_attrs.total_size,
                  Base_::tetrahedronattributelist);
      PrintDebug(debug, "set tetrahedronattributelist.");
    }

    auto tet_vols = ArrayInput<REAL>::from_object(tetrahedron_constraints,
                                                  "tetrahedron_constraints");
    if (!tet_vols.empty()) {
      tet_vols.check_shape({Base_::numberoftetrahedra, 1},
                           "tetrahedron_constraints");

      Base_::tetrahedronvolumelist = new REAL[tet_vols.total_size];
      std::copy_n(tet_vols.data,
                  tet_vols.total_size,
                  Base_::tetrahedronvolumelist);
      PrintDebug(debug, "set tetrahedronvolumelist.");
    }

    auto ref_elems =
        ArrayInput<int>::from_object(refine_elements, "refine_elements");
    if (!ref_elems.empty()) {
      ref_elems.check_shape({-1, n_input_tet_corners_}, "refine_elements");
      Base_::numberofrefineelems = static_cast<int>(ref_elems.shape[0]);
      PrintDebug(debug, "set numberofrefineelems:", Base_::numberofrefineelems);

      auto ref_vols = ArrayInput<REAL>::from_object(
          refine_element_constraints, "refine_element_constraints");
      ref_vols.check_shape({Base_::numberofrefineelems, 1},
                           "refine_element_constraints");

      Base_::refine_elem_list = new int[ref_elems.total_size];
      std::copy_n(ref_elems.data,
                  ref_elems.total_size,
                  Base_::refine_elem_list);
      PrintDebug(debug, "set refine_elem_list.");

      Base_::refine_elem_vol_list = new REAL[Base_::numberofrefineelems];
      std::copy_n(ref_vols.data,
                  Base_::numberofrefineelems,
                  Base_::refine_elem_vol_list);
      PrintDebug(debug, "set refine_elem_vol_list.");
    }

    auto trifaces_buf = ArrayInput<int>::from_object(trifaces, "trifaces");
    if (!trifaces_buf.empty()) {
      trifaces_buf.check_shape({-1, n_triface_corners_}, "trifaces");
      Base_::numberoftrifaces = static_cast<int>(trifaces_buf.shape[0]);
      PrintDebug(debug, "set numberoftrifaces:", Base_::numberoftrifaces);
      Base_::trifacelist = new int[trifaces_buf.total_size];
      std::copy_n(trifaces_buf.data,
                  trifaces_buf.total_size,
                  Base_::trifacelist);
      PrintDebug(debug, "set trifacelist.");

      auto tf_markers =
          ArrayInput<int>::from_object(triface_markers, "triface_markers");
      if (!tf_markers.empty()) {
        tf_markers.check_shape({Base_::numberoftrifaces, 1},
                               "triface_markers");
        Base_::trifacemarkerlist = new int[tf_markers.total_size];
        std::copy_n(tf_markers.data,
                    tf_markers.total_size,
                    Base_::trifacemarkerlist);
        PrintDebug(debug, "set trifacemarkerlist.");
      }
    }

    auto edges_buf = ArrayInput<int>::from_object(edges, "edges");
    if (!edges_buf.empty()) {
      edges_buf.check_shape({-1, n_edge_corners_}, "edges");
      Base_::numberofedges = static_cast<int>(edges_buf.shape[0]);
      PrintDebug(debug, "set numberofedges:", Base_::numberofedges);

      Base_::edgelist = new int[edges_buf.total_size];
      std::copy_n(edges_buf.data, edges_buf.total_size, Base_::edgelist);
      PrintDebug(debug, "set edgelist.");

      auto e_markers =
          ArrayInput<int>::from_object(edge_markers, "edge_markers");
      if (!e_markers.empty()) {
        e_markers.check_shape({Base_::numberofedges, 1}, "edge_markers");

        Base_::edgemarkerlist = new int[e_markers.total_size];
        std::copy_n(e_markers.data,
                    e_markers.total_size,
                    Base_::edgemarkerlist);
        PrintDebug(debug, "set edgemarkerlist");
      }
    }
  }

  /* getters */
  template<typename DataType>
  nb::ndarray<nb::numpy, DataType> CopyFromBase(
      const int base_data_count,
      const int base_data_stride,
      const DataType* base_array_ptr) {

    if (base_data_count <= 0 || base_array_ptr == nullptr ||
        base_data_stride <= 0) {
      DataType* dummy = new DataType[1]();
      nb::capsule owner(dummy, [](void* p) noexcept {
        delete[] static_cast<DataType*>(p);
      });
      size_t shape[2] = {0,
                         static_cast<size_t>(std::max(0, base_data_stride))};
      return nb::ndarray<nb::numpy, DataType>(dummy, 2, shape, owner);
    }

    const size_t total_len = static_cast<size_t>(base_data_count) *
                             static_cast<size_t>(base_data_stride);
    DataType* data = new DataType[total_len];
    std::copy_n(base_array_ptr, total_len, data);

    nb::capsule owner(data, [](void* p) noexcept {
      delete[] static_cast<DataType*>(p);
    });

    size_t shape[2] = {static_cast<size_t>(base_data_count),
                       static_cast<size_t>(base_data_stride)};
    return nb::ndarray<nb::numpy, DataType>(data, 2, shape, owner);
  }

  nb::ndarray<nb::numpy, REAL> GetPoints() {
    return CopyFromBase(Base_::numberofpoints, dim_, Base_::pointlist);
  }

  nb::ndarray<nb::numpy, REAL> GetPointAttributes() {
    return CopyFromBase(Base_::numberofpoints,
                        Base_::numberofpointattributes,
                        Base_::pointattributelist);
  }

  nb::ndarray<nb::numpy, int> GetTetrahedra() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        Base_::numberofcorners,
                        Base_::tetrahedronlist);
  }

  nb::ndarray<nb::numpy, REAL> GetTetrahedronAttributes() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        Base_::numberoftetrahedronattributes,
                        Base_::tetrahedronattributelist);
  }

  nb::ndarray<nb::numpy, int> GetTriFaces() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_triface_corners_,
                        Base_::trifacelist);
  }

  nb::ndarray<nb::numpy, int> GetTriFaceMarkers() {
    return CopyFromBase(Base_::numberoftrifaces, 1, Base_::trifacemarkerlist);
  }

  nb::ndarray<nb::numpy, int> GetO2Faces() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_triface_additional_second_order_nodes_,
                        Base_::o2facelist);
  }

  nb::ndarray<nb::numpy, int> GetNeighbors() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_tet_neighbors_,
                        Base_::neighborlist);
  }

  nb::ndarray<nb::numpy, int> GetTet2Faces() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_faces_per_tet_,
                        Base_::tet2facelist);
  }

  nb::ndarray<nb::numpy, int> GetTet2Edges() {
    return CopyFromBase(Base_::numberoftetrahedra,
                        n_edges_per_tet_,
                        Base_::tet2edgelist);
  }

  nb::ndarray<nb::numpy, int> GetFace2Tets() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_tet_neighbors_per_face_,
                        Base_::face2tetlist);
  }

  nb::ndarray<nb::numpy, int> GetFace2Edges() {
    return CopyFromBase(Base_::numberoftrifaces,
                        n_edges_per_face_,
                        Base_::face2edgelist);
  }

  nb::ndarray<nb::numpy, int> GetEdges() {
    return CopyFromBase(Base_::numberofedges, n_edge_corners_, Base_::edgelist);
  }

  nb::ndarray<nb::numpy, int> GetEdgeMarkers() {
    return CopyFromBase(Base_::numberofedges, 1, Base_::edgemarkerlist);
  }

  nb::ndarray<nb::numpy, int> GetO2Edges() {
    return CopyFromBase(Base_::numberofedges,
                        n_edge_additional_second_order_nodes_,
                        Base_::o2edgelist);
  }

  nb::ndarray<nb::numpy, int> GetEdge2Tets() {
    return CopyFromBase(Base_::numberofedges,
                        n_tet_per_edge_,
                        Base_::edge2tetlist);
  }

  nb::dict GetVoronoi() {
    nb::dict voronoi;

    voronoi["points"] =
        CopyFromBase(Base_::numberofvpoints, dim_, Base_::vpointlist);

    const size_t n_vedges = static_cast<size_t>(Base_::numberofvedges);
    const size_t vedges_total =
        n_vedges * static_cast<size_t>(n_voroedge_corners_);
    int* vedges_ptr = nullptr;
    if (vedges_total > 0) {
      vedges_ptr = new int[vedges_total]();
    } else {
      vedges_ptr = new int[1]();
    }

    int infinite_vertex_id{-1};
    nb::list edges_with_infinite_vertex;
    nb::list ray_directions;
    for (int i{}; i < Base_::numberofvedges; ++i) {
      const auto& base_voroedge = Base_::vedgelist[i];

      const int ve_id = i * n_voroedge_corners_;
      vedges_ptr[ve_id] = base_voroedge.v1;

      const auto& v2 = base_voroedge.v2;
      if (v2 < 0) {
        vedges_ptr[ve_id + 1] = infinite_vertex_id;
        --infinite_vertex_id;

        edges_with_infinite_vertex.append(i);

        nb::list normal;
        normal.append(base_voroedge.vnormal[0]);
        normal.append(base_voroedge.vnormal[1]);
        normal.append(base_voroedge.vnormal[2]);
        ray_directions.append(normal);
      } else {
        vedges_ptr[ve_id + 1] = v2;
      }
    }
    nb::capsule vedges_owner(vedges_ptr, [](void* p) noexcept {
      delete[] static_cast<int*>(p);
    });
    size_t vedges_shape[2] = {n_vedges,
                              static_cast<size_t>(n_voroedge_corners_)};
    voronoi["edges"] =
        nb::ndarray<nb::numpy, int>(vedges_ptr, 2, vedges_shape, vedges_owner);
    voronoi["edges_with_infinite_vertex"] =
        nb::make_tuple(edges_with_infinite_vertex, ray_directions);

    // facets
    const size_t n_vfacets = static_cast<size_t>(Base_::numberofvfacets);
    const size_t f2c_total =
        n_vfacets * static_cast<size_t>(n_cell_neighbors_per_vorofacet_);
    int* f2c_ptr = nullptr;
    if (f2c_total > 0) {
      f2c_ptr = new int[f2c_total]();
    } else {
      f2c_ptr = new int[1]();
    }

    nb::list facets;
    for (int i{}; i < Base_::numberofvfacets; ++i) {
      const auto& v_facet = Base_::vfacetlist[i];
      const int f2c_id = i * n_cell_neighbors_per_vorofacet_;
      f2c_ptr[f2c_id] = v_facet.c1;
      f2c_ptr[f2c_id + 1] = v_facet.c2;

      nb::list facet;
      for (int j{1}; j < v_facet.elist[0] + 1; ++j) {
        facet.append(v_facet.elist[j]);
      }
      facets.append(facet);
    }
    nb::capsule f2c_owner(f2c_ptr, [](void* p) noexcept {
      delete[] static_cast<int*>(p);
    });
    size_t f2c_shape[2] = {n_vfacets,
                           static_cast<size_t>(n_cell_neighbors_per_vorofacet_)};
    voronoi["facet2cell"] =
        nb::ndarray<nb::numpy, int>(f2c_ptr, 2, f2c_shape, f2c_owner);
    voronoi["facets"] = facets;

    // cells
    nb::list cells;
    for (int i{}; i < Base_::numberofvcells; ++i) {
      const auto& v_cell = Base_::vcelllist[i];
      nb::list cell;
      for (int j{1}; j < v_cell[0] + 1; ++j) {
        cell.append(v_cell[j]);
      }
      cells.append(cell);
    }
    voronoi["cells"] = cells;

    return voronoi;
  }

  // load from files
  void LoadNode(std::string fbase) { Base_::load_node(ToCStr(fbase).data()); }
  void LoadEdge(std::string fbase) { Base_::load_edge(ToCStr(fbase).data()); }
  void LoadFace(std::string fbase) { Base_::load_face(ToCStr(fbase).data()); }
  void LoadTet(std::string fbase) { Base_::load_tet(ToCStr(fbase).data()); }
  void LoadVol(std::string fbase) { Base_::load_vol(ToCStr(fbase).data()); }
  void LoadVar(std::string fbase) { Base_::load_var(ToCStr(fbase).data()); }
  void LoadMtr(std::string fbase) { Base_::load_mtr(ToCStr(fbase).data()); }
  void LoadElem(std::string fbase) { Base_::load_elem(ToCStr(fbase).data()); }
  void LoadPoly(std::string fbase) { Base_::load_poly(ToCStr(fbase).data()); }
  void LoadOff(std::string fbase) { Base_::load_off(ToCStr(fbase).data()); }
  void LoadPly(std::string fbase) { Base_::load_ply(ToCStr(fbase).data()); }
  void LoadStl(std::string fbase) { Base_::load_stl(ToCStr(fbase).data()); }
  void LoadVtk(std::string fbase) { Base_::load_vtk(ToCStr(fbase).data()); }
  void LoadMedit(std::string fbase, bool istetmesh) {
    Base_::load_medit(ToCStr(fbase).data(), (istetmesh) ? 1 : 0);
  }

  // save to files
  void SaveNodes(std::string fbase) { Base_::save_nodes(ToCStr(fbase).data()); }
  void SaveElements(std::string fbase) {
    Base_::save_elements(ToCStr(fbase).data());
  }
  void SaveFaces(std::string fbase) { Base_::save_faces(ToCStr(fbase).data()); }
  void SaveEdges(std::string fbase) { Base_::save_edges(ToCStr(fbase).data()); }
  void SaveNeighbors(std::string fbase) {
    Base_::save_neighbors(ToCStr(fbase).data());
  }
  void SavePoly(std::string fbase) { Base_::save_poly(ToCStr(fbase).data()); }
  void SaveFaces2Smesh(std::string fbase) {
    Base_::save_faces2smesh(ToCStr(fbase).data());
  }
};

inline void add_pytetgenio_class(nb::module_& m) {

  nb::class_<PyTetgenIo> klasse(m, "TetgenIO");

  klasse.def(nb::init<>())
      .def(nb::init<nb::handle,
                    nb::list,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::list,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    nb::handle,
                    bool>(),
           nb::arg("points"),
           nb::arg("facets"),
           nb::arg("point_attributes") = nb::none(),
           nb::arg("point_metrics") = nb::none(),
           nb::arg("facet_markers") = nb::none(),
           nb::arg("h_facets") = nb::list(),
           nb::arg("holes") = nb::none(),
           nb::arg("regions") = nb::none(),
           nb::arg("facet_constraints") = nb::none(),
           nb::arg("segment_constraints") = nb::none(),
           nb::arg("debug") = false)
      .def("setup_points",
           &PyTetgenIo::SetupPoints,
           nb::arg("points"),
           nb::arg("point_attributes") = nb::none(),
           nb::arg("point_metrics") = nb::none(),
           nb::arg("debug") = false)
      .def("setup_plc",
           &PyTetgenIo::SetupPlc,
           nb::arg("points"),
           nb::arg("facets"),
           nb::arg("point_attributes") = nb::none(),
           nb::arg("point_metrics") = nb::none(),
           nb::arg("facet_markers") = nb::none(),
           nb::arg("h_facets") = nb::list(),
           nb::arg("holes") = nb::none(),
           nb::arg("regions") = nb::none(),
           nb::arg("facet_constraints") = nb::none(),
           nb::arg("segment_constraints") = nb::none(),
           nb::arg("debug") = false)
      .def("setup_tetmesh",
           &PyTetgenIo::SetupTetMesh,
           nb::arg("points"),
           nb::arg("tetrahedra"),
           nb::arg("point_attributes") = nb::none(),
           nb::arg("point_metrics") = nb::none(),
           nb::arg("tetrahedron_attributes") = nb::none(),
           nb::arg("tetrahedron_constraints") = nb::none(),
           nb::arg("refine_elements") = nb::none(),
           nb::arg("refine_element_constraints") = nb::none(),
           nb::arg("trifaces") = nb::none(),
           nb::arg("triface_markers") = nb::none(),
           nb::arg("edges") = nb::none(),
           nb::arg("edge_markers") = nb::none(),
           nb::arg("debug") = false)
      .def("points", &PyTetgenIo::GetPoints)
      .def("pointattributes", &PyTetgenIo::GetPointAttributes)
      .def("tetrahedra", &PyTetgenIo::GetTetrahedra)
      .def("tetrahedronattributes", &PyTetgenIo::GetTetrahedronAttributes)
      .def("trifaces", &PyTetgenIo::GetTriFaces)
      .def("trifacemarkers", &PyTetgenIo::GetTriFaceMarkers)
      .def("o2faces", &PyTetgenIo::GetO2Faces)
      .def("neighbors", &PyTetgenIo::GetNeighbors)
      .def("tet2faces", &PyTetgenIo::GetTet2Faces)
      .def("tet2edges", &PyTetgenIo::GetTet2Edges)
      .def("face2tets", &PyTetgenIo::GetFace2Tets)
      .def("face2edges", &PyTetgenIo::GetFace2Edges)
      .def("edges", &PyTetgenIo::GetEdges)
      .def("edgemarkers", &PyTetgenIo::GetEdgeMarkers)
      .def("o2edges", &PyTetgenIo::GetO2Edges)
      .def("edge2tets", &PyTetgenIo::GetEdge2Tets)
      .def("voronoi", &PyTetgenIo::GetVoronoi)
      .def("load_node", &PyTetgenIo::LoadNode, nb::arg("fname_base"))
      .def("load_edge", &PyTetgenIo::LoadEdge, nb::arg("fname_base"))
      .def("load_face", &PyTetgenIo::LoadFace, nb::arg("fname_base"))
      .def("load_tet", &PyTetgenIo::LoadTet, nb::arg("fname_base"))
      .def("load_vol", &PyTetgenIo::LoadVol, nb::arg("fname_base"))
      .def("load_var", &PyTetgenIo::LoadVar, nb::arg("fname_base"))
      .def("load_mtr", &PyTetgenIo::LoadMtr, nb::arg("fname_base"))
      .def("load_elem", &PyTetgenIo::LoadElem, nb::arg("fname_base"))
      .def("load_poly", &PyTetgenIo::LoadPoly, nb::arg("fname_base"))
      .def("load_off", &PyTetgenIo::LoadOff, nb::arg("fname_base"))
      .def("load_ply", &PyTetgenIo::LoadPly, nb::arg("fname_base"))
      .def("load_stl", &PyTetgenIo::LoadStl, nb::arg("fname_base"))
      .def("load_vtk", &PyTetgenIo::LoadVtk, nb::arg("fname_base"))
      .def("load_medit",
           &PyTetgenIo::LoadMedit,
           nb::arg("fname_base"),
           nb::arg("is_tet_mesh"))
      .def("save_nodes", &PyTetgenIo::SaveNodes, nb::arg("fname_base"))
      .def("save_elements", &PyTetgenIo::SaveElements, nb::arg("fname_base"))
      .def("save_faces", &PyTetgenIo::SaveFaces, nb::arg("fname_base"))
      .def("save_edges", &PyTetgenIo::SaveEdges, nb::arg("fname_base"))
      .def("save_neighbors", &PyTetgenIo::SaveNeighbors, nb::arg("fname_base"))
      .def("save_poly", &PyTetgenIo::SavePoly, nb::arg("fname_base"))
      .def("save_faces2smesh",
           &PyTetgenIo::SaveFaces2Smesh,
           nb::arg("fname_base"));
}

} // namespace tetgenpy
