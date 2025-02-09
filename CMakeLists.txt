cmake_minimum_required(VERSION 3.22)
project(mekil)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MKL
find_package(MKL CONFIG REQUIRED PATHS /opt/intel/oneapi/mkl/latest/)
link_libraries(MKL::MKL)
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# add test
include_directories(include)
file(GLOB list ${CMAKE_CURRENT_LIST_DIR}/test/*.cpp)
message(STATUS "add executable: ${list}")
foreach(cpp IN LISTS list)
  get_filename_component(base_name ${cpp} NAME_WLE)
  message(${base_name})
  add_executable(${base_name} ${cpp})
endforeach()

# cmake package
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")
set(MEKIL_INCLUDE_INSTALL_DIR include/mekil)
install(FILES ${HEADERS} DESTINATION ${MEKIL_INCLUDE_INSTALL_DIR})

set(PACKAGE_VERSION "1.0.0")
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/mekil-config.cmake.in"
    "${PROJECT_BINARY_DIR}/mekil-config.cmake"
    @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/mekil-config.cmake"
        DESTINATION lib/cmake/mekil)
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/mekil-config-version.cmake.in"
    "${PROJECT_BINARY_DIR}/mekil-config-version.cmake"
    @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/mekil-config-version.cmake"
        DESTINATION lib/cmake/mekil)