cmake_minimum_required(VERSION 2.6)
project(Application)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(app app.cpp ctrl_ip.cpp general.cpp inst.cpp pre_image.cpp)
target_link_libraries(app -lfmt ${OpenCV_LIBS} -O3 -flto -march=native)

