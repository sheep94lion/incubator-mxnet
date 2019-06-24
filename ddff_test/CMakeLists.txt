# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
SET(CMAKE_C_STANDARD 11)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_BUILD_TYPE Debug)
# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_executable( # Sets the name of the library.
        native-lib

        # Provides a relative path to your source file(s).
        native-lib.cpp)

add_executable(native-lib-resnet native-lib-resnet.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

add_library( mxnet
             SHARED
             IMPORTED )

set_target_properties( # Specifies the target library
                       mxnet

                       # Specifies the parameter you want to define.
                       PROPERTIES IMPORTED_LOCATION

                       # Provides the path to the library you want to import.
                       /home/yizhao/Code/mxnet-dev/lib/libmxnet.so )

find_package(OpenCV REQUIRED)



# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        native-lib

        # Links the target library to the log library
        # included in the NDK.
        mxnet
        ${OpenCV_LIBS})

target_link_libraries(native-lib-resnet mxnet ${OpenCV_LIBS})

include_directories( /home/yizhao/Code/mxnet-dev/include )