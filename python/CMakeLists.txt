###############################################################################
#
# CMake build scripts for SPDlib python binding
#
# Created 2011/03/06 by Peter Bunting
#
# History
# 2011/03/06 - Created by Peter Bunting
#
###############################################################################

###############################################################################
# Set Project name and version
project (SPDPY)

# The version number.
set (SPDPY_VERSION_MAJOR 1)
set (SPDPY_VERSION_MINOR 1)
set (SPDPY_VERSION_PATCH 0)
set (SPDPY_VERSION "${SPDPY_VERSION_MAJOR}.${SPDPY_VERSION_MINOR}.${SPDPY_VERSION_PATCH}")
set (SPDPY_PACKAGE_VERSION "${SPDPY_VERSION_MAJOR}.${SPDPY_VERSION_MINOR}.${SPDPY_VERSION_PATCH}")
set (SPDPY_PACKAGE_STRING "SPDPY ${SPDPY_VERSION_MAJOR}.${SPDPY_VERSION_MINOR}.${SPDPY_VERSION_PATCH}")
set (SPDPY_PACKAGE_BUGREPORT "spdlib-develop@lists.sourceforge.net")
set (SPDPY_COPYRIGHT_YEAR 2012)
set (SPDPY_PACKAGE "SPDPY")

set(SPDLIB_IO_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for SPDLibIO")
set(SPDLIB_IO_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for SPDLibIO")

set(HDF5_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for HDF5")
set(HDF5_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for HDF5")

set(SPDPY_PY_MODULE_NAME spdpy)
###############################################################################

###############################################################################
# set file locations

set(SPDPY_H_Python
    spdpy_common.h
    spdpy_spdwriter.h
    spdpy_updwriter.h)

set(SPDPY_CPP_Python
    spdpy.cpp)

###############################################################################

###############################################################################
# CMake settings
cmake_minimum_required(VERSION 2.8.0)

IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE "DEBUG")
  SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

set(CMAKE_COLOR_MAKEFILE ON)

# Allow advanced users to generate Makefiles printing detailed commands
mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

# Path to additional CMake modules
set(CMAKE_MODULE_PATH ./cmake/modules ${CMAKE_MODULE_PATH})
###############################################################################


###############################################################################
# Platform and compiler specific settings

if(WIN32)
    if (MSVC)
        if (MSVC80 OR MSVC90 OR MSVC10)
            add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
            add_definitions(-D_CRT_SECURE_NO_WARNINGS)
            add_definitions(-D_CRT_NONSTDC_NO_WARNING)
            add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        endif()

        if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
            string(REGEX REPLACE "/W[0-4]" "/W2"
                   CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W2")
        endif()

        # Generate dot-user file with user-specific settings for Visual Studio project
        set(MSVC_ENVIRONMENT_PATH "" CACHE STRING
            "Custom PATH for Environment property in Visual Studio project configuration")
        mark_as_advanced(MSVC_ENVIRONMENT_PATH)
        set(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG "${ENVIRONMENT_PATH}")
        mark_as_advanced(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG)

    endif()

else()
  # Recommended C++ compilation flags
  # -Weffc++
  set(SPDLIB_COMMON_CXX_FLAGS
    "-ansi -Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wredundant-decls -Wno-long-long")
    #"-pedantic -ansi -Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wfloat-equal -Wredundant-decls -Wno-long-long")

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${SPDLIB_COMMON_CXX_FLAGS}")

  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPDLIB_COMMON_CXX_FLAGS}")
  endif()
endif(WIN32)

if (MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX .pyd)
else()
  set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
endif(MSVC)
###############################################################################

###############################################################################
# Check the required libraries are present
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

FIND_PACKAGE(PythonInterp) # so we can get PYTHON_VERSION_MAJOR for boost below
IF(NOT PYTHONINTERP_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find Python")
ENDIF()

FIND_PACKAGE(Boost)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")
  MESSAGE("Python version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  SET(SPD_BOOST_PYTHON_NAME PYTHON${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})

  SET(SPD_BOOST_PYTHON_NAME_LOWER python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
  MESSAGE("Searching for ${SPD_BOOST_PYTHON_NAME_LOWER}")
  FIND_PACKAGE(Boost COMPONENTS ${SPD_BOOST_PYTHON_NAME_LOWER})
  IF(${Boost_${SPD_BOOST_PYTHON_NAME}_FOUND})
    MESSAGE("FOUND BoostPython: ${Boost_${SPD_BOOST_PYTHON_NAME}_LIBRARY}")
  ELSE()
    MESSAGE(FATAL_ERROR "Unable to find correct Boost ${SPD_BOOST_PYTHON_NAME} version. Did you set BOOST_ROOT?")
  ENDIF()

ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

FIND_PACKAGE(PythonLibs)
IF(PYTHONLIBS_FOUND)
    INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIRS}")
    MESSAGE("FOUND Python Includes: ${PYTHON_INCLUDE_DIRS}")
	MESSAGE("FOUND Python Library: ${PYTHON_LIBRARIES}")
ELSEIF(NOT PYTHONLIBS_FOUND)
    MESSAGE(FATAL_ERROR "Unable to find python path")
ENDIF()

include_directories(${SPDLIB_IO_INCLUDE_DIR})
if (MSVC)
    set(SPDLIB_LIBRARY -LIBPATH:${SPDLIB_IO_LIB_PATH} libspdio.lib)
else()
    set(SPDLIB_LIBRARY -L${SPDLIB_IO_LIB_PATH} -lspdio )
endif(MSVC) 

include_directories(${HDF5_INCLUDE_DIR})
if (MSVC)
    set(HDF5_LIBRARIES -LIBPATH:${HDF5_LIB_PATH} hdf5.lib hdf5_hl.lib hdf5_cpp.lib)
else()
    set(HDF5_LIBRARIES -L${HDF5_LIB_PATH} -lhdf5 -lhdf5_hl -lhdf5_cpp)
endif(MSVC)

find_package( GDAL 1.6.0 REQUIRED )
if (GDAL_FOUND)
	include_directories (${GDAL_INCLUDE_DIR})
endif(GDAL_FOUND)

###############################################################################

###############################################################################
# Group source files for IDE source explorers
source_group("CMake Files" FILES CMakeLists.txt)
source_group("src_spd_python_cpp" FILES ${SPDPY_CPP_Python})
source_group("src_spd_python_h" FILES ${SPDPY_H_Python})
###############################################################################

###############################################################################
# Build and link library
add_library(${SPDPY_PY_MODULE_NAME} SHARED ${SPDPY_CPP_Python} ${SPDPY_H_Python})
target_link_libraries(${SPDPY_PY_MODULE_NAME} ${Boost_LIBRARIES} ${SPDLIB_LIBRARY} ${PYTHON_LIBRARIES} ${GDAL_LIBRARY})
SET_TARGET_PROPERTIES(${SPDPY_PY_MODULE_NAME} PROPERTIES PREFIX "")
###############################################################################

###############################################################################
# Installation
install (TARGETS ${SPDPY_PY_MODULE_NAME} DESTINATION site-packages)
###############################################################################
