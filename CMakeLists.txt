cmake_minimum_required(VERSION 3.15.7)
project(SSLAM)

set(CMAKE_VERBOSE_MAKEFILE OFF)

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("-- Build type: " ${CMAKE_BUILD_TYPE})

# Pass options to g++
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -w")
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++11.")

# Pass options to NVCC
#set(CMAKE_CXX_COMPILER "nvcc")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_50,code=sm_50")

# MaskRCNN config
set(MASKRCNN_DIR ${CMAKE_SOURCE_DIR}/src CACHE STRING "Location of MaskRCNN")
set(MASKRCNN_BENCHMARK_DIR ${CMAKE_SOURCE_DIR}/Thirdparty/maskrcnn-benchmark CACHE STRING "Location of MaskRCNN")
configure_file("${CMAKE_SOURCE_DIR}/src/MaskRCNN.py.in" "${CMAKE_CURRENT_BINARY_DIR}/MaskRCNN.py")

# TSDFfusion config
set(TSDF_DIR ${CMAKE_SOURCE_DIR}/src CACHE STRING "Location of TSDF")
set(TSDF_FUSION_PYTHON_DIR ${CMAKE_SOURCE_DIR}/Thirdparty/tsdf-fusion-python CACHE STRING "Location of tsdf-fusion-python")
configure_file("${CMAKE_SOURCE_DIR}/src/TSDFfusion.py.in" "${CMAKE_CURRENT_BINARY_DIR}/TSDFfusion.py")

# Dependencies
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV)
if(NOT OpenCV_FOUND)
	message(FATAL_ERROR "OpenCV 3.3.1 not found.")
else()
	message("-- Found OpenCV: ${OpenCV_INCLUDE_DIRS} (ver ${OpenCV_VERSION})")
endif()

set(Python3_USE_STATIC_LIBS FALSE)
find_package (Python3 COMPONENTS Interpreter Development NumPy)
if (Python3_Interpreter_FOUND)
	message("-- Found Python Includes: ${Python3_INCLUDE_DIRS}")
	message("-- Found Python Numpy: ${Python3_NumPy_INCLUDE_DIRS} (ver ${Python3_NumPy_VERSION})")
	message("-- Found Python Libraries: ${Python3_LIBRARIES}")
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)
message("-- Found CUDA: ${CUDA_INCLUDE_DIRS}")
message("-- Found CUDA: ${CUDA_LIBRARIES}")

find_package(Boost COMPONENTS program_options filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
add_definitions(${Boost_COMPONENTS_DEFINITIONS})

find_package(PCL REQUIRED) # 1.8
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${Python3_INCLUDE_DIRS}
	${Python3_NumPy_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/Thirdparty/ORB_SLAM2/ # to include ORB_SLAM2
	${CUDA_INCLUDE_DIRS}
)

set(ORB_SLAM2_LIBS ${CMAKE_SOURCE_DIR}/Thirdparty/ORB_SLAM2/lib/libORB_SLAM2.so)
set(DBoW2_LIBS ${CMAKE_SOURCE_DIR}/Thirdparty/ORB_SLAM2/Thirdparty/DBoW2/lib/libDBoW2.so)

# Compile libSSLAM
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
cuda_add_library(${PROJECT_NAME} SHARED
	src/Engine.cpp
	src/Inventory.cpp
	src/MaskRCNN.cpp
	src/InstanceViewer.cpp
	src/Object.cpp
	src/ObjectDrawer.cpp
	src/ObjectPoint.cpp
	src/TSDFfusion.cpp
	src/tsdf.cu
	src/DoN.cpp
	src/GroundRemoval.cpp
	src/Utility.cpp
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${ORB_SLAM2_LIBS}
	${DBoW2_LIBS}
	${Pangolin_LIBRARIES} # it should not need Pangolin here, its already in ${ORB_SLAM2_LIBS}
	${Python3_LIBRARIES}
	${CUDA_LIBRARIES}
)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build examples

#add_executable(label_instance ${CMAKE_SOURCE_DIR}/examples/label_instance.cpp src/MaskRCNN.py)
#set_property(TARGET label_instance 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(label_instance ${PROJECT_NAME})

# rgbd_tum
add_executable(rgbd_tum ${CMAKE_SOURCE_DIR}/examples/rgbd_tum.cpp)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# stereo_kitti
add_executable(stereo_kitti ${CMAKE_SOURCE_DIR}/examples/stereo_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

# label_instance_rgbd

#add_compile_options(-std=c++14)

add_executable(label_instance_rgbd ${CMAKE_SOURCE_DIR}/examples/label_instance_rgbd.cpp src/MaskRCNN.py src/TSDFfusion.py)
set_property(TARGET label_instance_rgbd
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(label_instance_rgbd ${PROJECT_NAME}  ${PCL_LIBRARIES})

# label_instance_lidar
add_executable(label_instance_lidar ${CMAKE_SOURCE_DIR}/examples/label_instance_lidar.cpp src/MaskRCNN.py src/TSDFfusion.py)
set_property(TARGET label_instance_lidar
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(label_instance_lidar ${PROJECT_NAME} ${Boost_LIBRARIES} ${PCL_LIBRARIES})


# Build tests
add_executable(testDoN ${CMAKE_SOURCE_DIR}/test/testDoN.cpp)
target_link_libraries(testDoN ${PROJECT_NAME} ${PCL_LIBRARIES})


