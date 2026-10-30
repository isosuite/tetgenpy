#include <nanobind/nanobind.h>

namespace tetgenpy {
namespace nb = nanobind;
void init_pytetgenio(nb::module_&);
void init_pytetrahedralize(nb::module_&);
} // namespace tetgenpy

NB_MODULE(tetgenpy_core, m) {
  tetgenpy::init_pytetgenio(m);
  tetgenpy::init_pytetrahedralize(m);
}
