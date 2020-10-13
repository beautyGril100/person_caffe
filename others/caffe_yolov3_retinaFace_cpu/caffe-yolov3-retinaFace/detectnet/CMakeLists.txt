
file(GLOB detectnetSources *.cpp)
file(GLOB detectnetIncludes *.h )

# cuda_add_executable(detectnet ${detectnetSources})
cuda_add_executable(detectnet detectnet.cpp)
target_link_libraries(detectnet yolov3-plugin)

cuda_add_executable(retinafaceNet retinafaceNet.cpp)
target_link_libraries(retinafaceNet yolov3-plugin)
