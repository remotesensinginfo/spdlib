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
set (SPDLIB_REPO_VERSION 309)
set (SPDLIB_VERSION_MAJOR 3)
set (SPDLIB_VERSION_MINOR 3)
set (SPDLIB_VERSION_PATCH 0)
set (SPDLIB_IO_VERSION_MAJOR 3)
set (SPDLIB_IO_VERSION_MINOR 0)
set (SPDLIB_IO_VERSION_PATCH 0)
set (SPDLIB_ALG_VERSION_MAJOR 3)
set (SPDLIB_ALG_VERSION_MINOR 8)
set (SPDLIB_ALG_VERSION_PATCH 0)
set (SPDLIB_NN_VERSION_MAJOR 1)
set (SPDLIB_NN_VERSION_MINOR 0)
set (SPDLIB_NN_VERSION_PATCH 0)

set (SPDLIB_VERSION "${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_VERSION_PATCH}")
set (SPDLIB_IO_VERSION "${SPDLIB_IO_VERSION_MAJOR}.${SPDLIB_IO_VERSION_MINOR}.${SPDLIB_IO_VERSION_PATCH}")
set (SPDLIB_ALG_VERSION "${SPDLIB_ALG_VERSION_MAJOR}.${SPDLIB_ALG_VERSION_MINOR}.${SPDLIB_ALG_VERSION_PATCH}")
set (SPDLIB_NN_VERSION "${SPDLIB_NN_VERSION_MAJOR}.${SPDLIB_NN_VERSION_MINOR}.${SPDLIB_NN_VERSION_PATCH}")
set (SPDLIB_PACKAGE_VERSION "${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_REPO_VERSION}")
set (SPDLIB_PACKAGE_STRING "SPDLib ${SPDLIB_VERSION_MAJOR}.${SPDLIB_VERSION_MINOR}.${SPDLIB_REPO_VERSION}")
set (SPDLIB_PACKAGE_BUGREPORT "spdlib-develop@lists.sourceforge.net")
set (SPDLIB_PACKAGE "SPDLib")
set (SPDLIB_COPYRIGHT_YEAR 2016)

if(WIN32)
    # Name of C++ library
    set(SPDLIB_LIB_NAME libspd)
	set(SPDLIB_MITLIB_NAME libspdio)
	set(SPDLIB_ALGLIB_NAME libspdalg)
	set(SPDLIB_NN_NAME libspdnn)
else()
    # Name of C++ library
    set(SPDLIB_LIB_NAME spd)
	set(SPDLIB_MITLIB_NAME spdio)
	set(SPDLIB_ALGLIB_NAME spdalg)
	set(SPDLIB_NN_NAME spdnn)
endif()

option (BUILD_SHARED_LIBS "Build with shared library" ON)
set(SPDLIB_WITH_UTILITIES TRUE CACHE BOOL "Choose if SPDLib utilities should be built")

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
configure_file ( "${PROJECT_HEADER_DIR}/spd-config.h.in" "${CMAKE_SOURCE_DIR}/${PROJECT_HEADER_DIR}/spd/spd-config.h" )
configure_file ( "${PROJECT_TOOLS_DIR}/spd-config.in" "${PROJECT_BINARY_DIR}/spd-config" )
configure_file ( "${PROJECT_TOOLS_DIR}/spdbatchgen.py.in" "${PROJECT_BINARY_DIR}/spdbatchgen.py" )
configure_file ( "${PROJECT_TOOLS_DIR}/spdbuildmergecmd.py.in" "${PROJECT_BINARY_DIR}/spdbuildmergecmd.py" )
configure_file ( "${PROJECT_TOOLS_DIR}/spdcmdgen.py.in" "${PROJECT_BINARY_DIR}/spdcmdgen.py" )
configure_file ( "${PROJECT_TOOLS_DIR}/spdbuildtileextractcmd.py.in" "${PROJECT_BINARY_DIR}/spdbuildtileextractcmd.py" )
configure_file ( "doc/Doxyfile.in" "doc/Doxyfile" )
configure_file ( "doc/dox_files/index.dox.in" "doc/dox_files/index.dox" )
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
        if (MSVC80 OR MSVC90 OR MSVC10 OR MSVC14)
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

        # Use cmake's MSVC_VERSION to work out the name of the Boost/CGAL DLL's 
        # to use (they use a different numbering scheme from cmake)
        set(MSVC_STRING_1500 "vc90")
        set(MSVC_STRING_1600 "vc100")
        set(MSVC_STRING_1700 "vc110")
        set(MSVC_STRING_1800 "vc120")
        set(MSVC_STRING_1900 "vc140")
        set(MSVC_STRING ${MSVC_STRING_${MSVC_VERSION}})
        message(STATUS "Using MSVC String " ${MSVC_STRING})
        
        # get GSL functions
        add_definitions(-DGSL_DLL)
        # HDF5 functions/classes
        add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB) 
        # this seems to be required otherwise OGRGeometry::IsEmpty is not found
        add_definitions(-DHAVE_GEOS)
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

if(APPLE)
    # use, i.e. don't skip the full RPATH for the build tree
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    
    # when building, don't use the install RPATH already
    # (but later on when installing)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
    
    # the RPATH to be used when installing
    SET(CMAKE_INSTALL_RPATH "")
    
    # don't add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
        
    SET(CMAKE_MACOSX_RPATH TRUE)
endif(APPLE)


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
if (MSVC)
    set(GDAL_LIBRARIES -LIBPATH:${GDAL_LIB_PATH} gdal_i.lib)
else()
    set(GDAL_LIBRARIES -L${GDAL_LIB_PATH} -lgdal)
endif(MSVC)

include_directories(${HDF5_INCLUDE_DIR})
if (MSVC)
    set(HDF5_LIBRARIES -LIBPATH:${HDF5_LIB_PATH} hdf5.lib hdf5_hl.lib hdf5_cpp.lib)
else()
    set(HDF5_LIBRARIES -L${HDF5_LIB_PATH} -lhdf5 -lhdf5_hl -lhdf5_cpp)
endif(MSVC)

include_directories(${BOOST_INCLUDE_DIR})

if (MSVC)
    # Under Windows, the boost libs are strangely named - use glob to find them
    file(GLOB BOOST_FS_LIB ${BOOST_LIB_PATH}/libboost_filesystem-${MSVC_STRING}-*.lib )
    if(NOT BOOST_FS_LIB)
        message(FATAL_ERROR "Could not find boost_filesystem lib")
    endif()
    file(GLOB BOOST_SYS_LIB ${BOOST_LIB_PATH}/libboost_system-${MSVC_STRING}-*.lib )
    if(NOT BOOST_SYS_LIB)
        message(FATAL_ERROR "Could not find boost_system lib")
    endif()
    message(STATUS "Found boost libraries " ${BOOST_FS_LIB} ${BOOST_SYS_LIB})
    set(BOOST_LIBRARIES -LIBPATH:${BOOST_LIB_PATH} ${BOOST_FS_LIB} ${BOOST_SYS_LIB})
else()
    set(BOOST_LIBRARIES -L${BOOST_LIB_PATH} -lboost_filesystem -lboost_system)
endif(MSVC)

include_directories(${GSL_INCLUDE_DIR})
if (MSVC)
    set(GSL_LIBRARIES -LIBPATH:${GSL_LIB_PATH} gsl.lib gslcblas.lib)
else()
    set(GSL_LIBRARIES -L${GSL_LIB_PATH} -lgsl -lgslcblas)
endif(MSVC)

include_directories(${LIBLAS_INCLUDE_DIR})
if (MSVC)
    set(LIBLAS_LIBRARIES -LIBPATH:${LIBLAS_LIB_PATH} las.lib)
else()
    set(LIBLAS_LIBRARIES -L${LIBLAS_LIB_PATH} -llas)
endif(MSVC)

include_directories(${CGAL_INCLUDE_DIR})
if (MSVC)
    file(GLOB CGAL_LIB_NAME ${BOOST_LIB_PATH}/CGAL-${MSVC_STRING}-*.lib )
    if(NOT CGAL_LIB_NAME)
        message(FATAL_ERROR "Could not find CGAL lib")
    endif()
    message(STATUS "Using CGAL lib " ${CGAL_LIB_NAME})
    set(CGAL_LIBRARIES -LIBPATH:${CGAL_LIB_PATH} ${CGAL_LIB_NAME})
else()
    set(CGAL_LIBRARIES -L${CGAL_LIB_PATH} -lCGAL)
endif(MSVC)

include_directories(${XERCESC_INCLUDE_DIR})
if (MSVC)
    set(XERCESC_LIBRARIES -LIBPATH:${XERCESC_LIB_PATH} xerces-c_3.lib)
else()
    set(XERCESC_LIBRARIES -L${XERCESC_LIB_PATH} -lxerces-c)
endif(MSVC)

include_directories(${GMP_INCLUDE_DIR})
if (MSVC)
    set(GMP_LIBRARIES -LIBPATH:${GMP_LIB_PATH} mpir.lib)
else()
    set(GMP_LIBRARIES -L${GMP_LIB_PATH} -lgmp)
endif(MSVC)

include_directories(${MPFR_INCLUDE_DIR})
if (MSVC)
    set(MPFR_LIBRARIES -LIBPATH:${MPFR_LIB_PATH} mpfr.lib)
else()
    set(MPFR_LIBRARIES -L${MPFR_LIB_PATH} -lmpfr)
endif(MSVC)
    
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
	add_executable(spdinfo ${PROJECT_SOURCE_DIR}/exe/spdinfo/main.cpp)
	target_link_libraries (spdinfo ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdthin ${PROJECT_SOURCE_DIR}/exe/spdthin/main.cpp)
	target_link_libraries (spdthin ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdextract ${PROJECT_SOURCE_DIR}/exe/spdextract/main.cpp)
	target_link_libraries (spdextract ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdwarp ${PROJECT_SOURCE_DIR}/exe/spdwarp/main.cpp)
	target_link_libraries (spdwarp ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdprofile ${PROJECT_SOURCE_DIR}/exe/spdprofile/main.cpp)
	target_link_libraries (spdprofile ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spddeftiles ${PROJECT_SOURCE_DIR}/exe/spddeftiles/main.cpp)
	target_link_libraries (spddeftiles ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdtileimg ${PROJECT_SOURCE_DIR}/exe/spdtileimg/main.cpp)
	target_link_libraries (spdtileimg ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdtiling ${PROJECT_SOURCE_DIR}/exe/spdtiling/main.cpp)
	target_link_libraries (spdtiling ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdgrdtidy ${PROJECT_SOURCE_DIR}/exe/spdgrdtidy/main.cpp)
	target_link_libraries (spdgrdtidy ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdclassify ${PROJECT_SOURCE_DIR}/exe/spdclassify/main.cpp)
	target_link_libraries (spdclassify ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
 	add_executable(spdtsample ${PROJECT_SOURCE_DIR}/exe/spdtsample/main.cpp)
	target_link_libraries (spdtsample ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
 	add_executable(spdsplit ${PROJECT_SOURCE_DIR}/exe/spdsplit/main.cpp)
	target_link_libraries (spdsplit ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )
	add_executable(spdclassimg ${PROJECT_SOURCE_DIR}/exe/spdclassimg/main.cpp)
	target_link_libraries (spdclassimg ${SPDLIB_MITLIB_NAME} ${SPDLIB_LIB_NAME} )  
endif(SPDLIB_WITH_UTILITIES)
###############################################################################

###############################################################################
# Installation
if (SPDLIB_WITH_UTILITIES)
	install (TARGETS spdtranslate spdversion spdcopy spdstats spdmaskgen spdpmfgrd spdmccgrd spdpolygrd spdinterp spddefheight spddecomp spdlastest spdmetrics spdmerge spdsubset spdoverlap spddefrgb spdelevation spdproj spdrmnoise spdclearclass spdpffgrd spdinfo spdthin spdextract spdwarp spdprofile spddeftiles spdtileimg spdtiling spdgrdtidy spdclassify spdtsample spdsplit spdclassimg DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif(SPDLIB_WITH_UTILITIES)

install (FILES "${CMAKE_BINARY_DIR}/${PROJECT_BINARY_DIR}/spd-config" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "${CMAKE_BINARY_DIR}/${PROJECT_BINARY_DIR}/spdbatchgen.py" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "${CMAKE_BINARY_DIR}/${PROJECT_BINARY_DIR}/spdbuildmergecmd.py" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "${CMAKE_BINARY_DIR}/${PROJECT_BINARY_DIR}/spdcmdgen.py" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "${CMAKE_BINARY_DIR}/${PROJECT_BINARY_DIR}/spdbuildtileextractcmd.py" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
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
