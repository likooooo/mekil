add_library(mekil SHARED cpu_backend.cpp)
set_target_properties(mekil PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) 
target_link_libraries(mekil PUBLIC ${MKL_IMPORTED_TARGETS})


file(GLOB tests ${CMAKE_CURRENT_LIST_DIR}/test_*.cpp)
foreach(test IN LISTS tests)
    get_filename_component(test_name ${test} NAME_WLE)
    add_executable(${test_name} ${test})
target_link_libraries(${test_name} PUBLIC mekil)
endforeach()

install(EXPORT mekilTargets
    FILE mekilTargets.cmake
    NAMESPACE mekil::
    DESTINATION cmake
)
install(TARGETS mekil EXPORT mekilTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)