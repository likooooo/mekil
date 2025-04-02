add_library(mekil SHARED cpu_backend.cpp)
set_target_properties(mekil PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) 
target_link_libraries(mekil PUBLIC ${MKL_IMPORTED_TARGETS})


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