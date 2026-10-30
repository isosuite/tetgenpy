# the tetgen fork declares an ancient cmake_minimum_required, which CMake>=4 rejects
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(TETGEN_BUILD_LIBRARY ON)
set(TETGEN_BUILD_EXECUTABLE OFF)
add_subdirectory(tetgen)
