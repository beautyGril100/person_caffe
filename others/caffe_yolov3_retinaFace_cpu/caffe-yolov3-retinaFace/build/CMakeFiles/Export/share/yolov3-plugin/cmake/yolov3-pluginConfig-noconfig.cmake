#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yolov3-plugin" for configuration ""
set_property(TARGET yolov3-plugin APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(yolov3-plugin PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "/usr/local/cuda/lib64/libcudart_static.a;-lpthread;dl;/usr/lib/x86_64-linux-gnu/librt.so;/home/hwzhu/ssd_new/caffe/build/lib/libcaffe.so;/usr/lib/x86_64-linux-gnu/libglog.so;/usr/lib/x86_64-linux-gnu/libgflags.so.2;/usr/lib/x86_64-linux-gnu/libboost_system.so;/usr/lib/x86_64-linux-gnu/libGLEW.so.1.13;opencv_stitching;opencv_core;opencv_videostab;opencv_calib3d;opencv_superres;opencv_highgui;opencv_photo;opencv_features2d;opencv_shape;opencv_ml;opencv_objdetect;opencv_imgcodecs;opencv_video;opencv_imgproc;opencv_dnn;opencv_videoio;opencv_flann"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/yolov3-plugin/libyolov3-plugin.so"
  IMPORTED_SONAME_NOCONFIG "libyolov3-plugin.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS yolov3-plugin )
list(APPEND _IMPORT_CHECK_FILES_FOR_yolov3-plugin "${_IMPORT_PREFIX}/lib/yolov3-plugin/libyolov3-plugin.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
