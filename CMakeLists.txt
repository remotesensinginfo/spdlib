###############################################################################
#
# CMake build scripts for SPDlib
# 
# Created 2010/12/14 by Peter Bunting
#
# These scripts were initial based on those used for libLAS (http://liblas.org/)
#
# History
# 2010/12/14 - Created by Peter Bunting
#
###############################################################################

###############################################################################
# Set Project name and version
project (SPDLIB)

#  Get the Subversion revision number for the project's source code.  The
#  VersionControl.cmake file tests the variable WINDOWS for the Windows
#  platform, so it needs to be set before including the file.
if (WIN32)
  set (WINDOWS ${WIN32})
endif()

set (PROJECT_BINARY_DIR bin)
set (PROJECT_LIBRARY_DIR lib)
set (PROJECT_SOURCE_DIR src)
set (PROJECT_HEADER_DIR include)
set (PROJECT_TOOLS_DIR tools)

# The version number.
set (SPDLIB_SVN_VERSION 509)
set (SPDLIB_VERSION_MAJOR 3)
set (SPDLIB_VERSION_MINOR 0)
set (SPDLIB_VERSION_PATCH 0)
set (SPDLIB_IO_VERSION_MAJOR 3)
set (SPDLIB_IO_VERSION_MINOR 0)
set (SPDLIB_IO_VERSION_PATCH 0)
set (SPDLIB_VERSION "${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_VERSION_PATCH}")
set (SPDLIB_IO_VERSION "${SPDLIB_IO_VERSION_MAJOR}.${SPDLIB_IO_VERSION_MINOR}.${SPDLIB_IO_VERSION_PATCH}")
set (SPDLIB_PACKAGE_VERSION "${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_VERSION_PATCH}")
set (SPDLIB_PACKAGE_STRING "SPDLib ${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_VERSION_PATCH}")
set (SPDLIB_PACKAGE_BUGREPORT "spdlib-develop@lists.sourceforge.net")
set (SPDLIB_PACKAGE "SPDLib")
set (SPDLIB_COPYRIGHT_YEAR 2012)

if(WIN32)
    # Name of C++ library
    set(SPDLIB_LIB_NAME libspd)
	set(SPDLIB_MITLIB_NAME libspdio)
else()
    # Name of C++ library
    set(SPDLIB_LIB_NAME spd)
	set(SPDLIB_MITLIB_NAME spdio)
endif()

option (BUILD_SHARED_LIBS "Build with shared library" ON)
set(SPDLIB_WITH_UTILITIES TRUE CACHE BOOL "Choose if SPDLib utilities should be built")

set(HDF5_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for HDF5")
set(HDF5_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for HDF5")

set(GSL_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for GSL")
set(GSL_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for GSL")

set(LIBLAS_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for LibLAS")
set(LIBLAS_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for LibLAS")

set(CGAL_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for CGAL")
set(CGAL_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for CGAL")

set(XERCESC_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for Xerces-c")
set(XERCESC_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for Xerces-c")

###############################################################################

###############################################################################
# CMake settings
cmake_minimum_required(VERSION 2.6.0)

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
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ./cmake/modules/)
###############################################################################

###############################################################################
# Setup configure file
configure_file ( "${PROJECT_HEADER_DIR}/spd-config.h.in" "${PROJECT_HEADER_DIR}/spd/spd-config.h" )
configure_file ( "${PROJECT_TOOLS_DIR}/spd-config.in" "${PROJECT_BINARY_DIR}/spd-config" )
configure_file ( "${PROJECT_TOOLS_DIR}/spdbatchgen.py.in" "${PROJECT_BINARY_DIR}/spdbatchgen.py" )
###############################################################################

###############################################################################
# Documentation
file(READ "doc/index.txt" README )
file(WRITE "README.txt" "${README}")
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
            string(REGEX REPLACE "/W[0-4]" "/W4"
                   CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        endif()

        # Generate dot-user file with user-specific settings for Visual Studio project
        set(MSVC_ENVIRONMENT_PATH "" CACHE STRING
            "Custom PATH for Environment property in Visual Studio project configuration")
        mark_as_advanced(MSVC_ENVIRONMENT_PATH)
        set(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG "${ENVIRONMENT_PATH}")
        mark_as_advanced(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG)

    endif()

	set(CMAKE_INCLUDE_PATH c:/osgeo4w/include;$ENV{CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH c:/osgeo4w/lib;$ENV{CMAKE_LIBRARY_PATH})
else()
  # Recommended C++ compilation flags
  # -Weffc++
  set(SPDLIB_COMMON_CXX_FLAGS
	"-Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wredundant-decls -Wno-long-long -DNDEBUG")
	#"-pedantic -ansi -Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wfloat-equal -Wredundant-decls -Wno-long-long")

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${SPDLIB_COMMON_CXX_FLAGS}")
    if (CMAKE_COMPILER_IS_GNUCXX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")
    endif()

  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPDLIB_COMMON_CXX_FLAGS}")
  endif()
endif(WIN32)

if (APPLE)
  set(SO_EXT dylib)
 # set(CMAKE_FIND_FRAMEWORK "LAST")
elseif(WIN32)
  set(SO_EXT dll)
else()
  set(SO_EXT so)
endif(APPLE)
###############################################################################

###############################################################################
# Check the required libraries are present

FIND_PACKAGE(Boost)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

find_package( GDAL 1.6.0 REQUIRED )
if (GDAL_FOUND)
	include_directories (${GDAL_INCLUDE_DIR})
endif(GDAL_FOUND)

include_directories(${HDF5_INCLUDE_DIR})
set(HDF5_LIBRARIES -L${HDF5_LIB_PATH} -lhdf5 -lhdf5_hl -lhdf5_cpp)

include_directories(${GSL_INCLUDE_DIR})
set(GSL_LIBRARIES -L${GSL_LIB_PATH} -lgsl -lgslcblas)

include_directories(${LIBLAS_INCLUDE_DIR})
set(LIBLAS_LIBRARY -L${LIBLAS_LIB_PATH} -llas -llas_c)

include_directories(${CGAL_INCLUDE_DIR})
set(CGAL_LIBRARY -L${CGAL_LIB_PATH} -lCGAL)

include_directories(${XERCESC_INCLUDE_DIR})
set(XERCESC_LIBRARIES -L${XERCESC_LIB_PATH} -lxerces-c)

###############################################################################

###############################################################################
# Build library

include_directories ("${PROJECT_HEADER_DIR}")
add_subdirectory ("${PROJECT_SOURCE_DIR}")
###############################################################################

###############################################################################
# Build executables
if (SPDLIB_WITH_UTILITIES)
	add_executable(spdtest ${PROJECT_SOURCE_DIR}/exe/test/main.cpp)
	target_link_libraries (spdtest ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdtranslate ${PROJECT_SOURCE_DIR}/exe/spdtranslate/main.cpp)
	target_link_libraries (spdtranslate ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdversion ${PROJECT_SOURCE_DIR}/exe/spdversion/main.cpp)
	target_link_libraries (spdversion ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdcopy ${PROJECT_SOURCE_DIR}/exe/spdcopy/main.cpp)
	target_link_libraries (spdcopy ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdstats ${PROJECT_SOURCE_DIR}/exe/spdstats/main.cpp)
	target_link_libraries (spdstats ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )	
	add_executable(spdmaskgen ${PROJECT_SOURCE_DIR}/exe/spdmaskgen/main.cpp)
	target_link_libraries (spdmaskgen ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdpmfgrd ${PROJECT_SOURCE_DIR}/exe/spdpmfgrd/main.cpp)
	target_link_libraries (spdpmfgrd ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdmccgrd ${PROJECT_SOURCE_DIR}/exe/spdmccgrd/main.cpp)
	target_link_libraries (spdmccgrd ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdpolygrd ${PROJECT_SOURCE_DIR}/exe/spdpolygrd/main.cpp)
	target_link_libraries (spdpolygrd ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdinterp ${PROJECT_SOURCE_DIR}/exe/spdinterp/main.cpp)
	target_link_libraries (spdinterp ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spddefheight ${PROJECT_SOURCE_DIR}/exe/spddefheight/main.cpp)
	target_link_libraries (spddefheight ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spddecomp ${PROJECT_SOURCE_DIR}/exe/spddecomp/main.cpp)
	target_link_libraries (spddecomp ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdlastest ${PROJECT_SOURCE_DIR}/exe/spdlastest/main.cpp)
	target_link_libraries (spdlastest ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdmetrics ${PROJECT_SOURCE_DIR}/exe/spdmetrics/main.cpp)
	target_link_libraries (spdmetrics ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdmerge ${PROJECT_SOURCE_DIR}/exe/spdmerge/main.cpp)
	target_link_libraries (spdmerge ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdsubset ${PROJECT_SOURCE_DIR}/exe/spdsubset/main.cpp)
	target_link_libraries (spdsubset ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdoverlap ${PROJECT_SOURCE_DIR}/exe/spdoverlap/main.cpp)
	target_link_libraries (spdoverlap ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdelevation ${PROJECT_SOURCE_DIR}/exe/spdelevation/main.cpp)
	target_link_libraries (spdelevation ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spddefrgb ${PROJECT_SOURCE_DIR}/exe/spddefrgb/main.cpp)
	target_link_libraries (spddefrgb ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdproj ${PROJECT_SOURCE_DIR}/exe/spdproj/main.cpp)
	target_link_libraries (spdproj ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdrmnoise ${PROJECT_SOURCE_DIR}/exe/spdrmnoise/main.cpp)
	target_link_libraries (spdrmnoise ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdclearclass ${PROJECT_SOURCE_DIR}/exe/spdclearclass/main.cpp)
	target_link_libraries (spdclearclass ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdpffgrd ${PROJECT_SOURCE_DIR}/exe/spdpffgrd/main.cpp)
	target_link_libraries (spdpffgrd ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
endif(SPDLIB_WITH_UTILITIES)
###############################################################################

###############################################################################
# Installation
if (SPDLIB_WITH_UTILITIES)
	install (TARGETS spdtranslate spdversion spdcopy spdstats spdmaskgen spdpmfgrd spdmccgrd spdpolygrd spdinterp spddefheight spddecomp spdlastest spdmetrics spdmerge spdsubset spdoverlap spddefrgb spdelevation spdproj spdrmnoise spdclearclass spdpffgrd DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif(SPDLIB_WITH_UTILITIES)

install (FILES "${PROJECT_BINARY_DIR}/spd-config" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "${PROJECT_BINARY_DIR}/spdbatchgen.py" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
###############################################################################

###############################################################################
# Create Distribution
SET(CPACK_SOURCE_GENERATOR "TGZ;ZIP;TBZ2")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")

set(CPACK_SOURCE_PACKAGE_FILE_NAME
  "${CMAKE_PROJECT_NAME}-${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_VERSION_PATCH}")

# Set files to ignore
list(APPEND CPACK_SOURCE_IGNORE_FILES "_CPack_Packages")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".gz")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".bz2")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".zip")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".svn")
list(APPEND CPACK_SOURCE_IGNORE_FILES "README")
list(APPEND CPACK_SOURCE_IGNORE_FILES "HOWTORELEASE.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CMakeCache.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CPackConfig.cmake")
list(APPEND CPACK_SOURCE_IGNORE_FILES "schemas")

include(CPack)

add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)
###############################################################################
