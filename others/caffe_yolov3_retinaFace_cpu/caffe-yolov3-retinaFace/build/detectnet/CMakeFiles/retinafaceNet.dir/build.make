# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build

# Include any dependencies generated for this target.
include detectnet/CMakeFiles/retinafaceNet.dir/depend.make

# Include the progress variables for this target.
include detectnet/CMakeFiles/retinafaceNet.dir/progress.make

# Include the compile flags for this target's objects.
include detectnet/CMakeFiles/retinafaceNet.dir/flags.make

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o: detectnet/CMakeFiles/retinafaceNet.dir/flags.make
detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o: ../detectnet/retinafaceNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o"
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o -c /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/detectnet/retinafaceNet.cpp

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.i"
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/detectnet/retinafaceNet.cpp > CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.i

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.s"
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/detectnet/retinafaceNet.cpp -o CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.s

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.requires:

.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.requires

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.provides: detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.requires
	$(MAKE) -f detectnet/CMakeFiles/retinafaceNet.dir/build.make detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.provides.build
.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.provides

detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.provides.build: detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o


# Object files for target retinafaceNet
retinafaceNet_OBJECTS = \
"CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o"

# External object files for target retinafaceNet
retinafaceNet_EXTERNAL_OBJECTS =

x86_64/bin/retinafaceNet: detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o
x86_64/bin/retinafaceNet: detectnet/CMakeFiles/retinafaceNet.dir/build.make
x86_64/bin/retinafaceNet: /usr/local/cuda/lib64/libcudart_static.a
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/librt.so
x86_64/bin/retinafaceNet: x86_64/lib/libyolov3-plugin.so
x86_64/bin/retinafaceNet: /usr/local/cuda/lib64/libcudart_static.a
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/librt.so
x86_64/bin/retinafaceNet: /home/hwzhu/ssd_new/caffe/build/lib/libcaffe.so
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/libglog.so
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/libgflags.so.2
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/libboost_system.so
x86_64/bin/retinafaceNet: /usr/lib/x86_64-linux-gnu/libGLEW.so.1.13
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_stitching.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_videostab.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_superres.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_photo.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_shape.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_ml.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_objdetect.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_calib3d.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_features2d.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_highgui.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_video.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_dnn.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_videoio.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_imgcodecs.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_imgproc.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_flann.so.3.4.5
x86_64/bin/retinafaceNet: /usr/local/lib/libopencv_core.so.3.4.5
x86_64/bin/retinafaceNet: detectnet/CMakeFiles/retinafaceNet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../x86_64/bin/retinafaceNet"
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/retinafaceNet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
detectnet/CMakeFiles/retinafaceNet.dir/build: x86_64/bin/retinafaceNet

.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/build

detectnet/CMakeFiles/retinafaceNet.dir/requires: detectnet/CMakeFiles/retinafaceNet.dir/retinafaceNet.cpp.o.requires

.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/requires

detectnet/CMakeFiles/retinafaceNet.dir/clean:
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet && $(CMAKE_COMMAND) -P CMakeFiles/retinafaceNet.dir/cmake_clean.cmake
.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/clean

detectnet/CMakeFiles/retinafaceNet.dir/depend:
	cd /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3 /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/detectnet /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet /workspace/hwzhu/caffe_yolov3_cpu/caffe-yolov3/build/detectnet/CMakeFiles/retinafaceNet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : detectnet/CMakeFiles/retinafaceNet.dir/depend

