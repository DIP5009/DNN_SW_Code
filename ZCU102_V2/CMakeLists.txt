cmake_minimum_required(VERSION 2.6)
project(Application)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(app app.cpp ctrl_ip.cpp general.cpp inst.cpp pre_image.cpp)
target_link_libraries(app -lfmt -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -Wall -flto -O3 -march=native)

