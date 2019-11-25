cmake_minimum_required(VERSION 3.5)
project(fedavg_mpi)

find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

add_executable(${PROJECT_NAME} main.c library.c)

target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${MPI_C_LIBRARIES})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -g -std=c99)
add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
