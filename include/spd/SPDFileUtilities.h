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

#include "spd/SPDException.h"

using namespace std;

namespace spdlib 
{
    class SPDFileUtilities
    {
    public: 
        SPDFileUtilities();
        static void getDIRList(string dir, list<string> *files) throw(SPDException);
        static void getDIRList(string dir, vector<string> *files) throw(SPDException);
        static void getDIRList(string dir, string ext, list<string> *files, bool withpath) throw(SPDException);
        static void getDIRList(string dir, string ext, vector<string> *files, bool withpath) throw(SPDException);
        static string* getDIRList(string dir, string ext,boost::uint_fast32_t *numFiles, bool withpath) throw(SPDException);
        static string* getFilesInDIRWithName(string dir, string name,boost::uint_fast32_t *numFiles) throw(SPDException);
        static string getFileNameNoExtension(string filepath);
        static string getFileName(string filepath);
        static string removeExtension(string filepath);
        static string getExtension(string filepath);
        static string getFileDirectoryPath(string filepath);
        static bool checkFilePresent(string file);
        static bool checkDIR4SHP(string dir, string shp) throw(SPDException);
        static void deleteSHP(string dir, string shp) throw(SPDException);
        ~SPDFileUtilities();
    };
}

#endif

