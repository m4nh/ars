cmake_minimum_required(VERSION 2.8.3)
project(roars)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  roscpp
  rospy
  cv_bridge
)

### Python wrapper
catkin_python_setup()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#############################################
## Eigen
#############################################
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#############################################
## OpenCV
#############################################
find_package(OpenCV REQUIRED)

#############################################
## Boost
#############################################
find_package(Boost 1.50.0 REQUIRED COMPONENTS)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR}) # cached
endif(Boost_FOUND)


#############################################
#############################################
## COMPILE DARKNET
#############################################
#############################################
option(USE_DARKNET off)
if(USE_DARKNET)

  #############################################
  ## CUDA
  #############################################
  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU -DCUDNN" )
  FIND_PACKAGE(CUDA)
  if (CUDA_FOUND)
      message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
      message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
      include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
      list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
      list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-Xcompiler \"-fPIC\" ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=compute_20 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=compute_30 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=compute_35 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=compute_50 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61 ")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_62,code=sm_62 ")
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      add_definitions(-DGPU)
      cuda_include_directories(src)
  else()
      list(APPEND LIBRARIES "m")
  endif()

  #############################################
  ## DARKNET
  #############################################
  set(DARKNET_FOLDER $ENV{DARKNET_PATH})
  MESSAGE(${DARKNET_FOLDER}) #DEBUG: export DARKNET_PATH=/home/daniele/work/workspace_cpp/darknet-cpp
  include_directories(
    ${DARKNET_FOLDER}/include
    ${DARKNET_FOLDER}/src
  )
  set(DARKNET_LIB ${DARKNET_FOLDER}/libdarknet-cpp-shared.so)
  ADD_LIBRARY(darknet SHARED IMPORTED)
  SET_TARGET_PROPERTIES(darknet PROPERTIES IMPORTED_LOCATION ${DARKNET_LIB})
endif()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES roars
#  CATKIN_DEPENDS image_transport roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

##########################################################################################
##########################################################################################
## NODES
##########################################################################################
##########################################################################################

if(USE_DARKNET)
  ## DARKNET RGB_SCAN DETECTOR 
  add_executable(darknet_rgb_detector src/nodes/darknet_rgb_detector.cpp)
  target_link_libraries(darknet_rgb_detector darknet ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

  ## DARKNET BATCH DETECTOR
  add_executable(darknet_batch_detector src/nodes/darknet_batch_detector.cpp)
  target_link_libraries(darknet_batch_detector darknet ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
endif()
