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
include tools/CMakeFiles/get_image_size.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/get_image_size.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/get_image_size.dir/flags.make

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o: tools/CMakeFiles/get_image_size.dir/flags.make
tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o: tools/get_image_size.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bwang/ssd_new/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o"
	cd /home/bwang/ssd_new/caffe/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/get_image_size.dir/get_image_size.cpp.o -c /home/bwang/ssd_new/caffe/tools/get_image_size.cpp

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_image_size.dir/get_image_size.cpp.i"
	cd /home/bwang/ssd_new/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bwang/ssd_new/caffe/tools/get_image_size.cpp > CMakeFiles/get_image_size.dir/get_image_size.cpp.i

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_image_size.dir/get_image_size.cpp.s"
	cd /home/bwang/ssd_new/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bwang/ssd_new/caffe/tools/get_image_size.cpp -o CMakeFiles/get_image_size.dir/get_image_size.cpp.s

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.requires:

.PHONY : tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.requires

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.provides: tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/get_image_size.dir/build.make tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.provides.build
.PHONY : tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.provides

tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.provides.build: tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o


# Object files for target get_image_size
get_image_size_OBJECTS = \
"CMakeFiles/get_image_size.dir/get_image_size.cpp.o"

# External object files for target get_image_size
get_image_size_EXTERNAL_OBJECTS =

tools/get_image_size: tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o
tools/get_image_size: tools/CMakeFiles/get_image_size.dir/build.make
tools/get_image_size: lib/libcaffe.so.1.0.0-rc3
tools/get_image_size: lib/libproto.a
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_regex.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libglog.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libsz.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libz.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libdl.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libm.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libglog.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libsz.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libz.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libdl.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libm.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/get_image_size: /usr/local/cuda/lib64/libcudart.so
tools/get_image_size: /usr/local/cuda/lib64/libcurand.so
tools/get_image_size: /usr/local/cuda/lib64/libcublas.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/get_image_size: /usr/local/lib/libopencv_highgui.so.3.3.0
tools/get_image_size: /usr/local/lib/libopencv_videoio.so.3.3.0
tools/get_image_size: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
tools/get_image_size: /usr/local/lib/libopencv_imgproc.so.3.3.0
tools/get_image_size: /usr/local/lib/libopencv_core.so.3.3.0
tools/get_image_size: /usr/local/lib/libopencv_cudev.so.3.3.0
tools/get_image_size: /usr/lib/liblapack.so
tools/get_image_size: /usr/lib/libcblas.so
tools/get_image_size: /usr/lib/libatlas.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/get_image_size: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/get_image_size: tools/CMakeFiles/get_image_size.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bwang/ssd_new/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable get_image_size"
	cd /home/bwang/ssd_new/caffe/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/get_image_size.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/get_image_size.dir/build: tools/get_image_size

.PHONY : tools/CMakeFiles/get_image_size.dir/build

tools/CMakeFiles/get_image_size.dir/requires: tools/CMakeFiles/get_image_size.dir/get_image_size.cpp.o.requires

.PHONY : tools/CMakeFiles/get_image_size.dir/requires

tools/CMakeFiles/get_image_size.dir/clean:
	cd /home/bwang/ssd_new/caffe/tools && $(CMAKE_COMMAND) -P CMakeFiles/get_image_size.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/get_image_size.dir/clean

tools/CMakeFiles/get_image_size.dir/depend:
	cd /home/bwang/ssd_new/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bwang/ssd_new/caffe /home/bwang/ssd_new/caffe/tools /home/bwang/ssd_new/caffe /home/bwang/ssd_new/caffe/tools /home/bwang/ssd_new/caffe/tools/CMakeFiles/get_image_size.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/get_image_size.dir/depend

