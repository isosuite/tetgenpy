#include <tetgenpy/py_tetrahedralize.hpp>

namespace tetgenpy {
namespace nb = nanobind;
void init_pytetrahedralize(nb::module_& m) { add_pytetrahedralize(m); }
} // namespace tetgenpy
