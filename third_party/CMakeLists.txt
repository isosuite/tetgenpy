# the tetgen fork declares an ancient cmake_minimum_required, which CMake>=4 rejects
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(TETGEN_BUILD_LIBRARY ON)
set(TETGEN_BUILD_EXECUTABLE OFF)
add_subdirectory(tetgen)
add_subdirectory(pybind11)
