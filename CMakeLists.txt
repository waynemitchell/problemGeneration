# problemGeneration library (using mfem and hypre)
# Author: Wayne Mitchell

cmake_minimum_required(VERSION 3.14)

# Declare the project and add library
project(problemGeneration
    VERSION 1.0.0
    LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_library(problemGeneration
   STATIC
   "problemGeneration.cpp"
   "optionsList.cpp"
   "visualization.cpp"
   "mfemtohypre.cpp"
   "getMesh.cpp"
   "diffusion.cpp"
   # "transport.cpp"
   "hypreMatrices.cpp")

# mfem
find_package(MFEM REQUIRED)
include_directories(${MFEM_INCLUDE_DIRS})
list(APPEND LIBRARIES "${MFEM_LIBRARY_DIR}/libmfem.a")

# metis
include_directories("${MY_PREFIX}/include")
list(APPEND LIBRARIES "${MY_PREFIX}/lib/libmetis.a")

# hypre
# find_package(HYPRE REQUIRED)
set(HYPRE_DIR "/home/nfs/wmitchell/Documents/hypre/src/hypre")
set(HYPRE_LIBRARY_DIR "${HYPRE_DIR}/lib")
set(HYPRE_INCLUDE_DIRS "${HYPRE_DIR}/include")
include_directories(${HYPRE_INCLUDE_DIRS})
list(APPEND LIBRARIES "${HYPRE_LIBRARY_DIR}/libHYPRE.a")

# OpenMP
# find_package(OpenMP)
# list(APPEND LIBRARIES OpenMP::OpenMP_CXX)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
list(APPEND LIBRARIES ${MPI_CXX_LIBRARIES})


# Link all libraries
target_link_libraries(problemGeneration ${LIBRARIES})

# Install
install(
   DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
   DESTINATION "include")
install(
    TARGETS problemGeneration
    EXPORT problemGenerationTargets
    ARCHIVE DESTINATION "lib"
    INCLUDES DESTINATION "include")
