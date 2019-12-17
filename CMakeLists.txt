# problemGeneration library (using mfem and hypre)
# Author: Wayne Mitchell

cmake_minimum_required(VERSION 3.14)

# Declare the project and add library
project(problemGeneration
    VERSION 1.0.0
    LANGUAGES CXX)

add_library(problemGeneration
   SHARED
   "problemGeneration.cpp"
   "optionsList.cpp"
   "mfemtohypre.cpp"
   "getMesh.cpp"
   "diffusion.cpp"
   "hypreMatrices.cpp")


# mfem
find_package(MFEM REQUIRED)
include_directories(SYSTEM ${MFEM_INCLUDE_DIRS})
# target_link_libraries(problemGeneration MFEM)



# CMakeshift stuff???
# find_package(CMakeshift 3.4 REQUIRED)
# include(CMakeshift/InstallSimpleLibraryPackage)
# cmakeshift_install_simple_library_package(
#     EXPORT problemGenerationTargets
#     VERSION_COMPATIBILITY SameMajorVersion
#     CONFIG_TEMPLATE "cmake/problemGenerationConfig.cmake.in")



# Install 
install(
   DIRECTORY "${PROJECT_SOURCE_DIR}/include/" 
   DESTINATION "include")
install(
    TARGETS problemGeneration
    EXPORT problemGenerationTargets
    ARCHIVE DESTINATION "lib"
    INCLUDES DESTINATION "include")