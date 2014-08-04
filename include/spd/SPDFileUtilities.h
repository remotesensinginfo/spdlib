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

#include <dirent.h>
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

namespace spdlib 
{
    class DllExport SPDFileUtilities
    {
    public: 
        SPDFileUtilities();
        static boost::uint_fast16_t getDIRCount(std::string dir) throw(SPDException);
        static void getDIRList(std::string dir, std::list<std::string> *files) throw(SPDException);
        static void getDIRList(std::string dir, std::vector<std::string> *files) throw(SPDException);
        static void getDIRList(std::string dir, std::string ext, std::list<std::string> *files, bool withpath) throw(SPDException);
        static void getDIRList(std::string dir, std::string ext, std::vector<std::string> *files, bool withpath) throw(SPDException);
        static std::string* getDIRList(std::string dir, std::string ext,boost::uint_fast32_t *numFiles, bool withpath) throw(SPDException);
        static std::string* getFilesInDIRWithName(std::string dir, std::string name,boost::uint_fast32_t *numFiles) throw(SPDException);
        static std::string getFileNameNoExtension(std::string filepath);
        static std::string getFileName(std::string filepath);
        static std::string removeExtension(std::string filepath);
        static std::string getExtension(std::string filepath);
        static std::string getFileDirectoryPath(std::string filepath);
        static bool checkFilePresent(std::string file);
        static bool checkDIR4SHP(std::string dir, std::string shp) throw(SPDException);
        static void deleteSHP(std::string dir, std::string shp) throw(SPDException);
        ~SPDFileUtilities();
    };
}

#endif

