#include <tetgenpy/py_tetgenio.hpp>

namespace tetgenpy {
namespace nb = nanobind;
void init_pytetgenio(nb::module_& m) { add_pytetgenio_class(m); }
} // namespace tetgenpy
