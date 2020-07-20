# problemGeneration library (using mfem and hypre)
# Author: Wayne Mitchell

cmake_minimum_required(VERSION 3.14)

# Declare the project and add library
project(problemGeneration
    VERSION 1.0.0
    LANGUAGES CXX)

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
list(APPEND LIBRARIES ${MFEM_LIBRARIES})

# hypre
find_package(HYPRE REQUIRED)
include_directories(${HYPRE_INCLUDE_DIRS})
list(APPEND LIBRARIES ${HYPRE_LIBRARIES})

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
