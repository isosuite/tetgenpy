find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(TETGENPY_SRCS init/py_tetgenio.cpp init/py_tetrahedralize.cpp
                  tetgenpy_core.cpp)

# add target
nanobind_add_module(tetgenpy_core STABLE_ABI NB_STATIC ${TETGENPY_SRCS})
target_include_directories(tetgenpy_core PRIVATE "${PROJECT_SOURCE_DIR}/cpp")
target_compile_definitions(tetgenpy_core PRIVATE ${TETGENPY_DEFS})
target_compile_options(tetgenpy_core PRIVATE ${TETGENPY_FLAGS}
                                             ${TETGENPY_OPTIMIZATION_FLAGS})
target_compile_features(tetgenpy_core PUBLIC cxx_std_17)

# link tetgen
target_link_libraries(tetgenpy_core PRIVATE tetgen)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/cpp/tetgenpy"
  DESTINATION ${incl_dest}
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  TARGETS tetgenpy_core
  DESTINATION tetgenpy
  COMPONENT PythonModule)
