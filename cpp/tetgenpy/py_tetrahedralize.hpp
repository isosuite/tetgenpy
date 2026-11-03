#pragma once

#include <string>
#include <vector>

#include <tetgen.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tetgenpy/py_tetgenio.hpp>

namespace tetgenpy {

namespace nb = nanobind;

/// Wrap tetrahedralize
inline void Tetrahedralize(std::string switches,
                           PyTetgenIo& in,
                           PyTetgenIo& out,
                           PyTetgenIo& additional_points,
                           PyTetgenIo& background_mesh) {

  // safely handle empty switches string
  std::vector<char> c_switches(switches.begin(), switches.end());
  c_switches.push_back('\0');

  // call tetrahedralize
  try {
    tetrahedralize(c_switches.data(),
                   &in,
                   &out,
                   &additional_points,
                   &background_mesh);
  } catch (int exit_code) {
    std::string msg;
    switch (exit_code) {
    case 1:
      msg = "TetGen error: Out of memory (exit code 1).";
      break;
    case 2:
      msg = "TetGen error: Encountered an internal error (exit code 2).";
      break;
    case 3:
      msg = "TetGen error: The input surface mesh contains self-intersections "
            "(exit code 3).";
      break;
    case 4:
      msg = "TetGen error: A very small input feature size was detected (exit "
            "code 4).";
      break;
    case 10:
      msg = "TetGen error: Invalid command line syntax or input file loading "
            "error (exit code 10).";
      break;
    default:
      msg = "TetGen error with exit code " + std::to_string(exit_code) + ".";
      break;
    }
    throw std::runtime_error(msg);
  }

  // mark output
  in.is_output_ = false;
  out.is_output_ = true;
  additional_points.is_output_ = false;
  background_mesh.is_output_ = false;
}

/// reproduce main(argc, argv)
inline int Main(nb::list sys_argv) {
  // prepare argc
  int argc = static_cast<int>(sys_argv.size());

  std::vector<std::string> argv_string;
  argv_string.reserve(argc);
  for (const auto& s_a : sys_argv) {
    argv_string.emplace_back(nb::cast<std::string>(s_a));
  }

  std::vector<char*> argv_char;
  argv_char.reserve(argc);
  for (auto& a_s : argv_string) {
    argv_char.emplace_back(a_s.data());
  }

  char** argv = argv_char.data();

  tetgenbehavior b;
  tetgenio in, addin, bgmin;

  try {
    if (!b.parse_commandline(argc, argv)) {
      terminatetetgen(NULL, 10);
    }

    // Read input files.
    if (b.refine) { // -r
      if (!in.load_tetmesh(b.infilename, (int) b.object)) {
        terminatetetgen(NULL, 10);
      }
    } else { // -p
      if (!in.load_plc(b.infilename, (int) b.object)) {
        terminatetetgen(NULL, 10);
      }
    }
    if (b.insertaddpoints) { // -i
      addin.load_node(b.addinfilename);
    }
    if (b.metric) { // -m
      bgmin.load_tetmesh(b.bgmeshfilename, (int) b.object);
    }

    tetrahedralize(&b, &in, NULL, &addin, &bgmin);
    return 0;
  } catch (int exit_code) {
    return exit_code;
  }
}

inline void add_pytetrahedralize(nb::module_& m) {
  m.def("tetrahedralize",
        &Tetrahedralize,
        nb::call_guard<nb::gil_scoped_release>(),
        nb::arg("switches"),
        nb::arg("in"),
        nb::arg("out"),
        nb::arg("additional_points"),
        nb::arg("background_mesh"));
  m.def("main", &Main, nb::arg("sys_argv"));
}

} // namespace tetgenpy
