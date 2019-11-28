include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXAMPLE "example")
add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
if (LIBARMA_PATH)
    target_link_libraries(${EXAMPLE} ${LIBARMA_PATH} lapack blas)
endif()

if (LIBBOOST_SYSTEM_MT_PATH AND LIBBOOST_THREAD_MT_PATH)
    target_link_libraries(${EXAMPLE} ${LIBBOOST_SYSTEM_MT_PATH} ${LIBBOOST_THREAD_MT_PATH})
endif()
