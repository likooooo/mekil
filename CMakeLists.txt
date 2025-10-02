cmake_minimum_required(VERSION 3.22)
project(mekil)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MKL
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# FFT
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(FFTW QUIET fftw3 IMPORTED_TARGET)
    pkg_search_module(FFTWF QUIET fftw3f IMPORTED_TARGET)
endif()

if(FFTW_FOUND)
    message(STATUS "FFTW found: ${FFTW_LIBRARIES}")
    add_compile_definitions(HAVE_FFTW)
    include_directories(PkgConfig::FFTW)
    link_libraries     (PkgConfig::FFTW)
    add_compile_definitions(FFTW_VERSION_STR=\"${FFTW_VERSION}\")
else()
    message(STATUS "FFTW not found, building without FFTW support")
endif()

if(FFTWF_FOUND)
    message(STATUS "FFTWF found: ${FFTWF_LIBRARIES}")
    add_compile_definitions(HAVE_FFTWF)
    include_directories(PkgConfig::FFTWF)
    link_libraries     (PkgConfig::FFTWF)
    add_compile_definitions(FFTWF_VERSION_STR=\"${FFTWF_VERSION}\")
else()
    message(STATUS "FFTWF not found, building without FFTWF support")
endif()

include_directories(include)
add_subdirectory(test)
add_subdirectory(src)

set(PACKAGE_VERSION "1.0.0")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")
set(MEKIL_INCLUDE_INSTALL_DIR include/mekil)
install(FILES ${HEADERS} DESTINATION ${MEKIL_INCLUDE_INSTALL_DIR})

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mekilConfig.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mekilConfig-version.cmake"
    DESTINATION cmake
)
include(${PROJECT_SOURCE_DIR}/cmake/auto_ctest.cmake)