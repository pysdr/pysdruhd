cmake_minimum_required(VERSION 3.7)
project(pysdruhd)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES pysdruhd.c usrp_wrapper.h)
add_library(pysdruhd SHARED ${SOURCE_FILES})

set_target_properties(pysdruhd PROPERTIES OUTPUT_NAME "pysdruhd" PREFIX "")

find_package(UHD)
# What does it take to support python 3 ???
find_package(PythonLibs 2.7 REQUIRED)

target_link_libraries(pysdruhd ${PYTHON_LIBRARY} ${UHD_LIBRARIES})
target_include_directories(pysdruhd PRIVATE ${PYTHON_INCLUDE_DIR} ${UHD_INCLUDE_DIRS})

install(TARGETS pysdruhd
        LIBRARY DESTINATION lib/python2.7/dist-packages)
