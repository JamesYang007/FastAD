include_directories(${PROJECT_SOURCE_DIR}/include)

# Check if armadillo include path exists
find_path_generic(ARMA_PATH "armadillo" "include")

# Check if boost include path exists
find_path_generic(BOOST_PATH "boost" "include")

# Check if armadillo library exists
find_path_generic(LIBARMA_PATH "armadillo" "library")

# Check if boost library exists
find_path_generic(LIBBOOST_SYSTEM_MT_PATH "boost_system-mt" "library")
find_path_generic(LIBBOOST_THREAD_MT_PATH "boost_thread-mt" "library")

set(EXAMPLE "example")
add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${EXAMPLE}
    ${LIBARMA_PATH}
    ${LIBBOOST_SYSTEM_MT_PATH}
    ${LIBBOOST_THREAD_MT_PATH}
    lapack
    blas
    )
