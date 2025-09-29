add_library(mekil SHARED cpu_backend.cpp)
set_target_properties(mekil PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) 
target_link_libraries(mekil PUBLIC ${MKL_IMPORTED_TARGETS})

file(GLOB tests ${CMAKE_CURRENT_LIST_DIR}/test_*.cpp)
# set(TOOLS_TARGETS "" CACHE INTERNAL "List of tool targets")
foreach(test IN LISTS tests)
    get_filename_component(test_name ${test} NAME_WLE)
    add_executable(${test_name} ${test})
    target_link_libraries(${test_name} PUBLIC mekil)
    # list(APPEND TOOLS_TARGETS ${test_name})
endforeach()
# copy_visualizer_files(mekil "$<TARGET_FILE_DIR:mekil>")

# add_custom_command(
#     TARGET mekil ${TOOLS_TARGETS} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             "${CMAKE_CURRENT_LIST_DIR}/cuda_test_fft_golden.py"
#             "$<TARGET_FILE_DIR:mekil>/cuda_test_fft_golden.py"
#     COMMENT "Copying cuda_test_fft_golden.py to $<TARGET_FILE_DIR:mekil>/cuda_test_fft_golden.py"
# )

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