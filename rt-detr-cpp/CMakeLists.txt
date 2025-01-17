cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(rt_detr_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------- OpenCV --------------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# -------------- LibTorch --------------
unset(TORCH_INSTALL_PREFIX CACHE)
unset(TORCH_FOUND CACHE)
unset(TORCH_LIBRARIES CACHE)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
message(STATUS "Torch library status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

# Add CURL and TIFF
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)

add_executable(inference src/main.cpp)
target_link_libraries(inference 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    CURL::libcurl
    TIFF::TIFF
)
set_property(TARGET inference PROPERTY CXX_STANDARD 17)

# -------------- Test Speed --------------
add_executable(test_speed src/test_speed.cpp)
target_link_libraries(test_speed ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET test_speed PROPERTY CXX_STANDARD 17)