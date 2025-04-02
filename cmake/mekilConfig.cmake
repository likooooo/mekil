# mekilConfig.cmake - Configuration file for the mekil package.

# Ensure that this script is included only once.
if(TARGET mekil)
    message(WARNING "NO mekil found")
    return()
endif()
message(STATUS "Found mekil")

# Get the directory where this file is located.
get_filename_component(_current_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)

# # Include the exported targets file.
include("${_current_dir}/mekilTargets.cmake")
# Set the package version variables.
set(mekil_VERSION_MAJOR 1) # Replace with your major version
set(mekil_VERSION_MINOR 0) # Replace with your minor version
set(mekil_VERSION_PATCH 0) # Replace with your patch version
set(mekil_VERSION "${mekil_VERSION_MAJOR}.${mekil_VERSION_MINOR}.${mekil_VERSION_PATCH}")

# Check if the requested version is compatible.
if(NOT "${mekil_FIND_VERSION}" STREQUAL "")
    if(NOT "${mekil_FIND_VERSION}" VERSION_LESS "${mekil_VERSION}")
        set(mekil_VERSION_COMPATIBLE TRUE)
    endif()

    if("${mekil_FIND_VERSION}" VERSION_EQUAL "${mekil_VERSION}")
        set(mekil_VERSION_EXACT TRUE)
    endif()
endif()

find_package(MKL CONFIG REQUIRED PATHS /opt/intel/oneapi/mkl/latest/)
# link_libraries(MKL::MKL)
link_libraries( ${MKL_IMPORTED_TARGETS})

message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

# Mark the package as found.
set(mekil_FOUND TRUE)