cmake_minimum_required(VERSION 3.10)
project( HomographyPlay )
find_package( OpenCV REQUIRED )
add_executable( crop_document src/crop_document.cpp )
target_link_libraries( crop_document ${OpenCV_LIBS} )
