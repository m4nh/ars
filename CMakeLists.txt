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








## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES roars
#  CATKIN_DEPENDS image_transport roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/roars.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/roars_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )



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



#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_roars.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
