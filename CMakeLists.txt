cmake_minimum_required(VERSION 3.0) # I don't actually know!
project(pysdruhd)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES pysdruhd.c usrp_wrapper.h wrapper_helper.c wrapper_helper.h usrp_object.h clock_and_time_sources.h sensors.h)
add_library(pysdruhd SHARED ${SOURCE_FILES})

set_target_properties(pysdruhd PROPERTIES OUTPUT_NAME "pysdruhd" PREFIX "" SUFFIX .so)

find_package(UHD)
# What does it take to support python 3 ???
find_package(PythonLibs 2.7 REQUIRED)
find_package(NumPy)

message("Using python lib ${PYTHON_LIBRARY}")
target_link_libraries(pysdruhd ${PYTHON_LIBRARY} ${UHD_LIBRARIES})
target_include_directories(pysdruhd PRIVATE ${PYTHON_INCLUDE_DIR} ${PYTHON_NUMPY_INCLUDE_DIR} ${UHD_INCLUDE_DIRS})

install(TARGETS pysdruhd
        LIBRARY DESTINATION lib/python2.7/site-packages)


#set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
#set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
#set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/module/__init__.py")
#set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
#
#configure_file(${SETUP_PY_IN} ${SETUP_PY})
#
#add_custom_command(OUTPUT ${OUTPUT}
#                   COMMAND ${PYTHON} ${SETUP_PY} build
#                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#                   DEPENDS ${DEPS})
#
#add_custom_target(target ALL DEPENDS ${OUTPUT})
#
#install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
