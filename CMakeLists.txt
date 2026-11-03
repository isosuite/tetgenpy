cmake_minimum_required(VERSION 3.15)
project(
  tetgenpy
  VERSION 0.1.0
  LANGUAGES CXX)

# config
set(exe_dest "bin")
set(incl_dest "include")
set(lib_dest "lib")
set(cfg_dest "${lib_dest}/cmake/${PROJECT_NAME}")
set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(version_config "${gen_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${gen_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# flags / defs
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(TETGENPY_FLAGS -fPIC)
  set(TETGENPY_OPTIMIZATION_FLAGS $<$<NOT:$<CONFIG:Debug>>:-O3>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(TETGENPY_FLAGS)
  set(TETGENPY_OPTIMIZATION_FLAGS $<$<NOT:$<CONFIG:Debug>>:/O2>)
endif()

set(TETGENPY_DEFS)

if(CMAKE_BUILD_TYPE MATCHES Debug)

  # compiler specific flags
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(TETGENPY_FLAGS ${TETGENPY_FLAGS} -Wall -Wextra -Wpedantic
                       -Wzero-as-null-pointer-constant -Wno-unused)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(TETGENPY_FLAGS
        ${TETGENPY_FLAGS}
        -Wall
        -Wextra
        -Wmost
        -Wextra
        -Wpedantic
        -Wunreachable-code
        -Wshadow
        -Wfloat-equal
        -Weffc++
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wzero-as-null-pointer-constant)
  else()
    message(WARNING "tetgenpy is only tested with GNU and Clang.")
  endif()
endif()

add_subdirectory(third_party)
add_subdirectory(cpp/tetgenpy)

# configure config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${version_config}"
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file("cmake/config.cmake.in" "${project_config}"
                              INSTALL_DESTINATION "${cfg_dest}")

if(TETGEN_BUILD_LIBRARY)
  install(
    TARGETS tetgen
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION ${lib_dest}
    ARCHIVE DESTINATION ${lib_dest}
    INCLUDES
    DESTINATION "${incl_dest}")
endif()

if(TETGEN_BUILD_EXECUTABLE)
  install(
    TARGETS tetgen_exec
    EXPORT ${TARGETS_EXPORT_NAME}
    RUNTIME DESTINATION ${exe_dest})
endif()

install(FILES "${project_config}" "${version_config}" DESTINATION "${cfg_dest}")

install(
  EXPORT "${TARGETS_EXPORT_NAME}"
  NAMESPACE "${namespace}"
  DESTINATION "${cfg_dest}")
install(FILES ${CXX_HEADERS} DESTINATION ${incl_dest})
