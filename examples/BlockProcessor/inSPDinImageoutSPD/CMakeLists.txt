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
project (SPDLIB_Example)

set(BOOST_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for Boost")
set(BOOST_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for Boost")

set(GDAL_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for GDAL")
set(GDAL_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for GDAL")

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

set(GMP_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for GMP")
set(GMP_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for GMP")

set(MPFR_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for MPFR")
set(MPFR_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MPFR")

set(SPDLIB_INCLUDE_DIR /usr/local/include CACHE PATH "Include PATH for SPDLIB")
set(SPDLIB_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for SPDLIB")

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

        # MSVC can't compile the whole of SPDLib - yet. Seems to do ok on spdio
        set(BUILD_SPDIO_ONLY ON)
        message(STATUS "MSVC detected. BUILD_SPDIO_ONLY set to ON.")
        
    endif()

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

include_directories(${GDAL_INCLUDE_DIR})
set(GDAL_LIBRARIES -L${GDAL_LIB_PATH} -lgdal)

include_directories(${HDF5_INCLUDE_DIR})
set(HDF5_LIBRARIES -L${HDF5_LIB_PATH} -lhdf5 -lhdf5_hl -lhdf5_cpp)

include_directories(${BOOST_INCLUDE_DIR})
set(BOOST_LIBRARIES -L${BOOST_LIB_PATH} -lboost_filesystem -lboost_system)

include_directories(${GSL_INCLUDE_DIR})
set(GSL_LIBRARIES -L${GSL_LIB_PATH} -lgsl -lgslcblas)

include_directories(${LIBLAS_INCLUDE_DIR})
set(LIBLAS_LIBRARIES -L${LIBLAS_LIB_PATH} -llas -llas_c)

include_directories(${CGAL_INCLUDE_DIR})
set(CGAL_LIBRARIES -L${CGAL_LIB_PATH} -lCGAL)

include_directories(${XERCESC_INCLUDE_DIR})
set(XERCESC_LIBRARIES -L${XERCESC_LIB_PATH} -lxerces-c)

include_directories(${GMP_INCLUDE_DIR})
set(GMP_LIBRARIES -L${GMP_LIB_PATH} -lgmp)

include_directories(${MPFR_INCLUDE_DIR})
set(MPFR_LIBRARIES -L${MPFR_LIB_PATH} -lmpfr)

include_directories(${SPDLIB_INCLUDE_DIR})
set(SPDLIB_LIBRARIES -L${SPDLIB_LIB_PATH} -lspdio -lspd)
###############################################################################


###############################################################################
# Build executables
add_executable(spdexample main.cpp)
target_link_libraries (spdexample ${SPDLIB_LIBRARIES} )

###############################################################################

###############################################################################
# Installation
install (TARGETS spdexample DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
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
