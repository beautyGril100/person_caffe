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


# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_SOURCE_DIR = /home/bwang/ssd_new/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bwang/ssd_new/caffe

# Include any dependencies generated for this target.
include examples/CMakeFiles/convert_mnist_siamese_data.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/convert_mnist_siamese_data.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/convert_mnist_siamese_data.dir/flags.make

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o: examples/CMakeFiles/convert_mnist_siamese_data.dir/flags.make
examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o: examples/siamese/convert_mnist_siamese_data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bwang/ssd_new/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o"
	cd /home/bwang/ssd_new/caffe/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o -c /home/bwang/ssd_new/caffe/examples/siamese/convert_mnist_siamese_data.cpp

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.i"
	cd /home/bwang/ssd_new/caffe/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bwang/ssd_new/caffe/examples/siamese/convert_mnist_siamese_data.cpp > CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.i

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.s"
	cd /home/bwang/ssd_new/caffe/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bwang/ssd_new/caffe/examples/siamese/convert_mnist_siamese_data.cpp -o CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.s

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.requires:

.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.requires

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.provides: examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/convert_mnist_siamese_data.dir/build.make examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.provides.build
.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.provides

examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.provides.build: examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o


# Object files for target convert_mnist_siamese_data
convert_mnist_siamese_data_OBJECTS = \
"CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o"

# External object files for target convert_mnist_siamese_data
convert_mnist_siamese_data_EXTERNAL_OBJECTS =

examples/siamese/convert_mnist_siamese_data: examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o
examples/siamese/convert_mnist_siamese_data: examples/CMakeFiles/convert_mnist_siamese_data.dir/build.make
examples/siamese/convert_mnist_siamese_data: lib/libcaffe.so.1.0.0-rc3
examples/siamese/convert_mnist_siamese_data: lib/libproto.a
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_regex.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libglog.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libsz.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libz.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libdl.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libm.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libglog.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libsz.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libz.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libdl.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libm.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/liblmdb.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libleveldb.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libsnappy.so
examples/siamese/convert_mnist_siamese_data: /usr/local/cuda/lib64/libcudart.so
examples/siamese/convert_mnist_siamese_data: /usr/local/cuda/lib64/libcurand.so
examples/siamese/convert_mnist_siamese_data: /usr/local/cuda/lib64/libcublas.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libcudnn.so
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_highgui.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_videoio.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_imgproc.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_core.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/local/lib/libopencv_cudev.so.3.3.0
examples/siamese/convert_mnist_siamese_data: /usr/lib/liblapack.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/libcblas.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/libatlas.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libpython2.7.so
examples/siamese/convert_mnist_siamese_data: /usr/lib/x86_64-linux-gnu/libboost_python.so
examples/siamese/convert_mnist_siamese_data: examples/CMakeFiles/convert_mnist_siamese_data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bwang/ssd_new/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable siamese/convert_mnist_siamese_data"
	cd /home/bwang/ssd_new/caffe/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_mnist_siamese_data.dir/link.txt --verbose=$(VERBOSE)
	cd /home/bwang/ssd_new/caffe/examples && ln -sf /home/bwang/ssd_new/caffe/examples/siamese/convert_mnist_siamese_data /home/bwang/ssd_new/caffe/examples/siamese/convert_mnist_siamese_data.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/convert_mnist_siamese_data.dir/build: examples/siamese/convert_mnist_siamese_data

.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/build

examples/CMakeFiles/convert_mnist_siamese_data.dir/requires: examples/CMakeFiles/convert_mnist_siamese_data.dir/siamese/convert_mnist_siamese_data.cpp.o.requires

.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/requires

examples/CMakeFiles/convert_mnist_siamese_data.dir/clean:
	cd /home/bwang/ssd_new/caffe/examples && $(CMAKE_COMMAND) -P CMakeFiles/convert_mnist_siamese_data.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/clean

examples/CMakeFiles/convert_mnist_siamese_data.dir/depend:
	cd /home/bwang/ssd_new/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bwang/ssd_new/caffe /home/bwang/ssd_new/caffe/examples /home/bwang/ssd_new/caffe /home/bwang/ssd_new/caffe/examples /home/bwang/ssd_new/caffe/examples/CMakeFiles/convert_mnist_siamese_data.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/convert_mnist_siamese_data.dir/depend

