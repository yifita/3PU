cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...without compiler extensions like gnu++11

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

find_package(Qt5Core)
find_package(Qt5Gui)
get_target_property(QtGui_location Qt5::Gui LOCATION)
message("Found Qt5Gui ${Qt5Gui_LIBRARIES} in ${QtGui_location}")
message("Found Qt5Gui include ${Qt5Gui_INCLUDE_DIRS}")
include_directories(../vcglib)
include_directories(../vcglib/eigenlib)
file(GLOB CPP_FILES *.cpp)
add_executable(polygonmesh ${CPP_FILES} ../vcglib/wrap/ply/plylib.cpp)
target_link_libraries(polygonmesh Qt5::Core ${Qt5Gui_LIBRARIES})
