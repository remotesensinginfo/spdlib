/*
 *  SPDFileUtilities.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 09/03/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  SPDLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SPDLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SPDFileUtilities_H
#define SPDFileUtilities_H

// dirent.h now included in the .cpp so we don't have to install
// Windows emulation header
#include <errno.h>
#include <vector>
#include <list>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <stdint.h>
#include <stdio.h>

#include <boost/cstdint.hpp>
#include "boost/filesystem.hpp"

#include "spd/SPDException.h"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspd_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib 
{
    class DllExport SPDFileUtilities
    {
    public: 
        SPDFileUtilities();
        static boost::uint_fast16_t getDIRCount(std::string dir) ;
        static void getDIRList(std::string dir, std::list<std::string> *files) ;
        static void getDIRList(std::string dir, std::vector<std::string> *files) ;
        static void getDIRList(std::string dir, std::string ext, std::list<std::string> *files, bool withpath) ;
        static void getDIRList(std::string dir, std::string ext, std::vector<std::string> *files, bool withpath) ;
        static std::string* getDIRList(std::string dir, std::string ext,boost::uint_fast32_t *numFiles, bool withpath) ;
        static std::string* getFilesInDIRWithName(std::string dir, std::string name,boost::uint_fast32_t *numFiles) ;
        static std::string getFileNameNoExtension(std::string filepath);
        static std::string getFileName(std::string filepath);
        static std::string removeExtension(std::string filepath);
        static std::string getExtension(std::string filepath);
        static std::string getFileDirectoryPath(std::string filepath);
        static bool checkFilePresent(std::string file);
        static bool checkDIR4SHP(std::string dir, std::string shp) ;
        static void deleteSHP(std::string dir, std::string shp) ;
        ~SPDFileUtilities();
    };
}

#endif

